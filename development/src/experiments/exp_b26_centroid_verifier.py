"""EXP-B26: add one train-only centroid view to the B25 locked three-view decoder.

C0 is the linear two-centroid control. C1 keeps up to four weighted prototypes
per class. Each outer fold chooses between them on inner OOF, then the outer
held-out questions score that choice against the fold-local three-view lock.
Standalone centroid scores are diagnostics. They do not decide the gate.
Validation stays closed when the fused system misses the internal gate.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b07_cross_attn import kfold_indices  # noqa: E402
from experiments.exp_b22_g05_r5 import (  # noqa: E402
    B20_STRICT_ALL,
    B20_STRICT_HARD,
    B20_STRICT_WEIGHTS,
    HISTORICAL_ALL,
    HISTORICAL_HARD,
    HISTORICAL_WEIGHTS,
    official_score,
    score_choice,
    system_choice,
    write_official_predictions,
)
from experiments.exp_b23_hard_weighting import complement_rows  # noqa: E402
from experiments.exp_b25_decoder_alignment import (  # noqa: E402
    jsonable,
    retention_decision,
    rows_sha256,
    score_held_config,
    sha256_file,
    write_json,
)
from experiments.multiseed import (  # noqa: E402
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    configure_determinism,
    environment_record,
    register_seeds,
    rel,
)
from models.centroid_verifier import ARMS, fit_centroids, score_centroids  # noqa: E402
from postprocessing.aligned_fusion import (  # noqa: E402
    build_cluster_table,
    fit_aligned_weights,
    internal_gate_decision,
    normalize_views,
    question_eligibility,
    score_aligned,
    select_config,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
N_FOLDS = 5
PUBLISHED_LOCKED_HARD = (353, 337, 297, 342, 358)
PUBLISHED_LOCKED_ALL = (860, 864, 856, 877, 871)
ALLOWED_SOURCE_DRIFT = {"dev/runs/audit_logs/seed_registry.json"}

EXP_DIR = RUNS / "exp_b26"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b26_centroid_audit.json"
CONTRACT_PATH = RUNS / "audit_logs" / "private_configuration_omitted.json"
B25_DIR = RUNS / "exp_b25"
B25_AUDIT_PATH = RUNS / "audit_logs" / "exp_b25_decoder_alignment_audit.json"


def slice_bank(bank: Sequence[Dict[str, Any]], rows: np.ndarray) -> List[Dict[str, Any]]:
    return [bank[int(row)] for row in rows]


def hits_of(choices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    chosen = np.asarray(choices, dtype=np.int64)
    return labels[np.arange(labels.shape[0]), chosen] >= 0.5


def flip_counts(new_hits: np.ndarray, old_hits: np.ndarray) -> Dict[str, int]:
    new_hits = np.asarray(new_hits, dtype=bool)
    old_hits = np.asarray(old_hits, dtype=bool)
    return {
        "corrected_old_errors": int((new_hits & ~old_hits).sum()),
        "spoiled_old_correct": int((~new_hits & old_hits).sum()),
        "both_correct": int((new_hits & old_hits).sum()),
        "both_wrong": int((~new_hits & ~old_hits).sum()),
    }


def correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> Optional[float]:
    first = np.asarray(left, dtype=np.float64)[mask]
    second = np.asarray(right, dtype=np.float64)[mask]
    if first.size < 2 or float(np.std(first)) == 0.0 or float(np.std(second)) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def check_sources(contract: Dict[str, Any]) -> List[Dict[str, str]]:
    drifted = []
    for relative, expected in contract["source_sha256"].items():
        actual = sha256_file(WORKSPACE / relative)
        if actual != expected:
            drifted.append({"file": relative, "expected": expected, "actual": actual})
    unexpected = [item for item in drifted if item["file"] not in ALLOWED_SOURCE_DRIFT]
    if unexpected:
        raise RuntimeError(f"unexpected source drift before B26: {unexpected}")
    return drifted


def load_b25_caches() -> Tuple[np.lib.npyio.NpzFile, Dict[str, Any], Dict[str, Any]]:
    b25_audit = json.loads(B25_AUDIT_PATH.read_text(encoding="utf-8"))
    if b25_audit.get("verdict") not in {"VALIDATION_NOT_RETAINED", "RETAINED"}:
        raise RuntimeError("B26 requires a B25 internal gate that passed")
    if not b25_audit["internal_gate"]["passed"]:
        raise RuntimeError("B25 internal gate did not pass; B26 must keep the historical decoder")
    oof_path = B25_DIR / "oof_logits.npz"
    lock_path = B25_DIR / "fold_local_locks.json"
    full_lock_path = B25_DIR / "downstream_decoder_lock.json"
    if sha256_file(oof_path) != b25_audit["oof_sha256"]:
        raise RuntimeError("B25 OOF file does not match its audit hash")
    if sha256_file(lock_path) != b25_audit["fold_local_lock_sha256"]:
        raise RuntimeError("B25 fold-local locks do not match the audit hash")
    if sha256_file(full_lock_path) != b25_audit["decoder_lock"]["sha256"]:
        raise RuntimeError("B25 full-train decoder lock does not match the audit hash")
    return np.load(oof_path), json.loads(lock_path.read_text(encoding="utf-8")), json.loads(full_lock_path.read_text(encoding="utf-8"))


def decode_held(
    config: Dict[str, Any],
    weights: Sequence[float],
    scales: Optional[Sequence[float]],
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if config["normalization"] == "fold_OOF_global_scale" and not scales:
        raise RuntimeError("held-out global scale is missing its fit-partition RMS")
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    normalized, _scales = normalize_views(views, config["normalization"], eligible, scales=scales)
    scored = score_aligned(
        normalized, labels, hard, bank, parses, eligible, table,
        weights, float(config["temperature"]), float(config["rho"]), np.arange(len(bank)),
    )
    choices = np.asarray(scored.pop("choices"), dtype=np.int64)
    return choices, scored


def fit_fusion(
    config: Dict[str, Any],
    views: Sequence[np.ndarray],
    hard: np.ndarray,
    table: Dict[str, np.ndarray],
    eligible: np.ndarray,
    fit_rows: np.ndarray,
) -> Tuple[np.ndarray, Optional[List[float]], Dict[str, Any], List[np.ndarray]]:
    normalized, scales = normalize_views(views, config["normalization"], eligible, rows=fit_rows)
    weights, info = fit_aligned_weights(
        normalized, hard, table, float(config["temperature"]), float(config["rho"]), fit_rows,
    )
    return weights, scales, info, normalized


def replay_locked_reference(
    oof: np.lib.npyio.NpzFile,
    locks: Dict[str, Any],
    folds: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """The B26 reference is the B25 fold-local three-view decoder, not the older T=1 fit."""
    replayed = {}
    for record in locks["folds"]:
        fold_id = int(record["outer_fold"])
        held = folds[fold_id]
        fit_rows = complement_rows(folds, [fold_id])
        if rows_sha256(held) != record["held_rows_sha256"]:
            raise RuntimeError(f"outer fold {fold_id} row hash does not match the B25 lock")
        if rows_sha256(fit_rows) != record["fit_rows_sha256"]:
            raise RuntimeError(f"outer fold {fold_id} fit-row hash does not match the B25 lock")
        held_bank = slice_bank(bank, held)
        views = [oof["full_mean_a"][held], oof["full_mean_b"][held], oof["full_tree"][held]]
        checked = score_held_config(
            record["selected_config"], record["weights"], record["scales"],
            views, labels[held], hard[held], held_bank,
        )
        choices, scored = decode_held(
            record["selected_config"], record["weights"], record["scales"],
            views, labels[held], hard[held], held_bank,
        )
        if checked["hard_correct"] != int(record["selected"]["hard_correct"]):
            raise RuntimeError(f"fold {fold_id} lock replay hard {checked['hard_correct']} drifted")
        if scored["hard_correct"] != checked["hard_correct"] or scored["all_correct"] != checked["all_correct"]:
            raise RuntimeError(f"fold {fold_id} decoder recount drifted")
        if checked["hard_correct"] != PUBLISHED_LOCKED_HARD[fold_id]:
            raise RuntimeError(f"fold {fold_id} lock hard is not the published B25 count")
        if checked["all_correct"] != PUBLISHED_LOCKED_ALL[fold_id]:
            raise RuntimeError(f"fold {fold_id} lock all-count is not the published B25 count")
        replayed[fold_id] = {"choices": choices, "counts": checked, "config": record["selected_config"], "scales": record["scales"]}
        print(
            f"  lock replay fold {fold_id}: {checked['hard_correct']}/{checked['hard_total']} "
            f"all {checked['all_correct']}/{checked['all_total']}",
            flush=True,
        )
    return replayed


def score_nested_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
    parses: Sequence[Sequence[Optional[str]]],
    hard: np.ndarray,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    count = len(labels)
    scores = {arm: np.zeros((N_FOLDS, count, 4), dtype=np.float64) for arm in ARMS}
    filled = np.zeros((N_FOLDS, count), dtype=bool)
    seconds = {arm: 0.0 for arm in ARMS}
    cell_id = 0
    for outer in range(N_FOLDS):
        for inner in range(N_FOLDS):
            if inner == outer:
                continue
            cell_id += 1
            fit_rows = complement_rows(folds, [outer, inner])
            held = folds[inner]
            if set(int(row) for row in fit_rows).intersection(int(row) for row in held):
                raise RuntimeError(f"centroid cell outer {outer} inner {inner} leaked")
            print(f"  nested centroid {cell_id}/20 outer {outer} inner {inner}", flush=True)
            for arm in ARMS:
                model = fit_centroids(arm, embeddings, labels, parses, hard, fit_rows)
                scores[arm][outer, held] = score_centroids(model, embeddings[held])
                seconds[arm] += float(model["fit_seconds"])
                print(
                    f"    {arm} K={model['actual_k']} tau={model['tau']} {model['fit_seconds']:.2f}s",
                    flush=True,
                )
            filled[outer, held] = True
    return {"scores": scores, "filled": filled, "seconds": seconds}


def score_holdout_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
    parses: Sequence[Sequence[Optional[str]]],
    hard: np.ndarray,
    folds: Sequence[np.ndarray],
    arm: str,
) -> Tuple[np.ndarray, float]:
    count = len(labels)
    scores = np.zeros((count, 4), dtype=np.float64)
    seconds = 0.0
    for fold_id, held in enumerate(folds):
        fit_rows = complement_rows(folds, [fold_id])
        model = fit_centroids(arm, embeddings, labels, parses, hard, fit_rows)
        scores[held] = score_centroids(model, embeddings[held])
        seconds += float(model["fit_seconds"])
    return scores, seconds


def select_arm(
    config: Dict[str, Any],
    base_views: Sequence[np.ndarray],
    geometry: Dict[str, np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    folds: Sequence[np.ndarray],
    excluded: Sequence[int],
) -> Dict[str, Any]:
    pooled = []
    standalone = []
    banned = {int(fold_id) for fold_id in excluded}
    inners = [fold_id for fold_id in range(N_FOLDS) if fold_id not in banned]
    for arm in ARMS:
        views = list(base_views) + [geometry[arm]]
        hard_correct = all_correct = hard_total = all_total = absent = 0
        standalone_hard = standalone_all = 0
        nll_sum = 0.0
        for inner in inners:
            fit_rows = complement_rows(folds, list(banned | {inner}))
            eval_rows = folds[inner]
            _weights, _scales, _info, normalized = fit_fusion(
                config, views, hard, table, eligible, fit_rows,
            )
            scored = score_aligned(
                normalized, labels, hard, bank, parses, eligible, table,
                _weights, float(config["temperature"]), float(config["rho"]), eval_rows,
            )
            hard_correct += int(scored["hard_correct"])
            hard_total += int(scored["hard_total"])
            all_correct += int(scored["all_correct"])
            all_total += int(scored["all_total"])
            nll_sum += float(scored["nll_sum"])
            absent += int(scored["absent_gold_clusters"])
            alone_norm, _alone_scales = normalize_views(
                [views[-1]], config["normalization"], eligible, rows=fit_rows,
            )
            alone = score_aligned(
                alone_norm, labels, hard, bank, parses, eligible, table,
                [1.0], float(config["temperature"]), float(config["rho"]), eval_rows,
            )
            standalone_hard += int(alone["hard_correct"])
            standalone_all += int(alone["all_correct"])
        pooled.append({
            "name": arm,
            "arm": arm,
            "hard_correct": hard_correct,
            "hard_total": hard_total,
            "all_correct": all_correct,
            "all_total": all_total,
            "nll_sum": nll_sum,
            "nll_count": all_total,
            "absent_gold_clusters": absent,
        })
        standalone.append({
            "arm": arm,
            "hard_correct": standalone_hard,
            "hard_total": hard_total,
            "all_correct": standalone_all,
            "all_total": all_total,
        })
    index = select_config(pooled)
    return {"arm": ARMS[index], "index": index, "pooled": pooled, "standalone": standalone}


def run_validation_confirmation(
    candidate: Dict[str, Any],
    train_pack: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """One frozen four-view candidate. The centroid model is already fit on train only."""
    val = SplitData("validation")
    val_hard = np.asarray(val.hard, dtype=bool)
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    tree_val = np.asarray(np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")["seed_42"])
    geometry_val = score_centroids(candidate["model"], val.emb)
    ensemble_views = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], tree_val, geometry_val]
    historical_views = ensemble_views[:3]
    historical = score_choice(system_choice(HISTORICAL_WEIGHTS, historical_views, val.bank), val.labels, val_hard)
    strict = score_choice(system_choice(B20_STRICT_WEIGHTS, historical_views, val.bank), val.labels, val_hard)
    if historical["val_hard"]["correct"] != HISTORICAL_HARD or historical["val_all"]["correct"] != HISTORICAL_ALL:
        raise RuntimeError("historical three-view replay missed 300/743")
    if strict["val_hard"]["correct"] != B20_STRICT_HARD or strict["val_all"]["correct"] != B20_STRICT_ALL:
        raise RuntimeError("strict three-view replay missed 295/738")

    config = candidate["config"]
    labels = train_pack["labels"]
    hard = train_pack["hard"]
    table = train_pack["table"]
    eligible = train_pack["eligible"]
    started = time.perf_counter()
    ensemble_choice, ensemble_score = decode_held(
        config, candidate["weights"], candidate["scales"], ensemble_views, val.labels, val_hard, val.bank,
    )
    decode_seconds = time.perf_counter() - started
    reference_weights, reference_scales, _info, _norm = fit_fusion(
        config,
        [train_pack["mean_a"], train_pack["mean_b"], train_pack["tree"]],
        hard, table, eligible, np.arange(labels.shape[0]),
    )
    reference_choice, _reference_score = decode_held(
        config, reference_weights, reference_scales, historical_views, val.labels, val_hard, val.bank,
    )
    reference_counts = {
        "hard_correct": int(hits_of(reference_choice, val.labels)[val_hard].sum()),
        "hard_total": int(val_hard.sum()),
        "all_correct": int(hits_of(reference_choice, val.labels).sum()),
        "all_total": int(len(val.bank)),
    }
    paired = []
    for seed in SEEDS:
        seed_views = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], tree_val]
        old_weights, old_scales, _old_info, _old_norm = fit_fusion(
            config,
            [train_pack["per_seed_a"][int(seed)], train_pack["per_seed_b"][int(seed)], train_pack["tree"]],
            hard, table, eligible, np.arange(labels.shape[0]),
        )
        old_choice, old_score = decode_held(
            config, old_weights, old_scales, seed_views, val.labels, val_hard, val.bank,
        )
        new_weights, new_scales, new_info, _new_norm = fit_fusion(
            config,
            [
                train_pack["per_seed_a"][int(seed)],
                train_pack["per_seed_b"][int(seed)],
                train_pack["tree"],
                train_pack["geometry_oof"],
            ],
            hard, table, eligible, np.arange(labels.shape[0]),
        )
        new_choice, new_score = decode_held(
            config, new_weights, new_scales, seed_views + [geometry_val], val.labels, val_hard, val.bank,
        )
        paired.append({
            "seed": int(seed),
            "old_hard": int(old_score["hard_correct"]),
            "new_hard": int(new_score["hard_correct"]),
            "hard_delta": int(new_score["hard_correct"] - old_score["hard_correct"]),
            "old_all": int(old_score["all_correct"]),
            "new_all": int(new_score["all_correct"]),
            "all_delta": int(new_score["all_correct"] - old_score["all_correct"]),
            "weights": [float(value) for value in new_weights],
            "scales": new_scales,
        })
        del old_choice, new_choice
    prediction_path = EXP_DIR / "exp_b26_validation_predictions.json"
    score_path = EXP_DIR / "exp_b26_validation_score.json"
    write_official_predictions(prediction_path, val.bank, ensemble_choice)
    official = official_score(prediction_path, score_path)
    if int(official["correct"]) != int(ensemble_score["all_correct"]):
        raise RuntimeError("official all-count does not match the in-process four-view decode")
    new_hits = hits_of(ensemble_choice, val.labels)
    old_hits = hits_of(reference_choice, val.labels)
    asset_bytes = int(contract["baseline"]["historical"]["recorded_uncompressed_bytes"])
    lock_bytes = (B25_DIR / "downstream_decoder_lock.json").stat().st_size
    centroid_bytes = (EXP_DIR / "centroid_model.npz").stat().st_size
    confirmation = {
        "historical_reproduction": {
            "val_hard": int(historical["val_hard"]["correct"]),
            "val_all": int(historical["val_all"]["correct"]),
        },
        "strict_reproduction": {
            "val_hard": int(strict["val_hard"]["correct"]),
            "val_all": int(strict["val_all"]["correct"]),
        },
        "ensemble": {
            "hard_correct": int(ensemble_score["hard_correct"]),
            "hard_total": int(ensemble_score["hard_total"]),
            "all_correct": int(ensemble_score["all_correct"]),
            "all_total": int(ensemble_score["all_total"]),
            "easy_correct": int(ensemble_score["all_correct"] - ensemble_score["hard_correct"]),
            "nll_sum": float(ensemble_score["nll_sum"]),
            "absent_gold_clusters": int(ensemble_score["absent_gold_clusters"]),
        },
        "same_protocol_reference": reference_counts,
        "paired": paired,
        "flips_vs_locked_three_view": flip_counts(new_hits, old_hits),
        "official_score": official,
        "in_process_decode_seconds": decode_seconds,
        "asset_preaudit": {
            "scope": "local in-process decode plus unchanged B20 assets and the centroid file",
            "not_a_fresh_predict_process": True,
            "not_a_xeon_certificate": True,
            "b20_uncompressed_bytes": asset_bytes,
            "b25_lock_bytes": int(lock_bytes),
            "centroid_bytes": int(centroid_bytes),
            "combined_bytes": int(asset_bytes + lock_bytes + centroid_bytes),
        },
        "validation_configurations": 4,
        "selected_arm": candidate["arm"],
    }
    confirmation["retention_gate"] = retention_decision(confirmation, contract)
    return confirmation


def run_exp_b26() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B26: centroid view on the locked decoder")
    print("=" * 78, flush=True)
    started = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "experiment": "EXP-B26",
        "protocol": PROTOCOL,
        "reference_system": "B25 fold-local locked three-view decoder",
        "validation_queries": 0,
        "validation_opened": False,
        "test_references_opened": False,
        "written_to_model_bundle": False,
        "candidate_submission_updated": False,
        "validation_confirmation": None,
        "verdict": "FAILED",
    }
    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        batch = contract["batches"]["EXP-B26"]
        if int(batch["max_unique_inner_configurations"]) != 2:
            raise RuntimeError("B26 config budget is not 2")
        if tuple(batch["arms"]) != ARMS:
            raise RuntimeError("B26 arm order is not C0 then C1")
        drifted = check_sources(contract)
        device = configure_determinism()
        environment = environment_record(device)
        environment["executable"] = sys.executable
        print(f"Interpreter: {sys.executable}", flush=True)
        oof, locks, full_lock = load_b25_caches()
        train = SplitData("train")
        labels = train.labels
        hard = np.asarray(train.hard, dtype=bool)
        if len(train.bank) != 4755 or int(hard.sum()) != 2114:
            raise RuntimeError("train split is not the registered 4755/2114 partition")
        folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
        eligible, parses = question_eligibility(train.bank)
        table = build_cluster_table(parses, eligible, labels)
        print("Replaying B25 fold-local three-view locks", flush=True)
        replayed = replay_locked_reference(oof, locks, folds, labels, hard, train.bank)

        print("Fitting nested C0/C1 centroids", flush=True)
        nested = score_nested_geometry(train.emb, labels, parses, hard, folds)
        reference_folds = []
        selected_folds = []
        fold_records = []
        pooled_flips = {"corrected_old_errors": 0, "spoiled_old_correct": 0, "both_correct": 0, "both_wrong": 0}
        pooled_arm = {"corrected_old_errors": 0, "spoiled_old_correct": 0, "both_correct": 0, "both_wrong": 0}
        correlations = []
        for record in locks["folds"]:
            fold_id = int(record["outer_fold"])
            held = folds[fold_id]
            complement = complement_rows(folds, [fold_id])
            if not bool(nested["filled"][fold_id, complement].all()):
                raise RuntimeError(f"outer {fold_id} is missing nested centroid scores")
            config = record["selected_config"]
            base_views = [oof["nested_mean_a"][fold_id], oof["nested_mean_b"][fold_id], oof["nested_tree"][fold_id]]
            print(f"Selecting centroid on outer {fold_id}", flush=True)
            selection = select_arm(
                config, base_views,
                {arm: nested["scores"][arm][fold_id] for arm in ARMS},
                labels, hard, train.bank, parses, eligible, table, folds, [fold_id],
            )
            arm = selection["arm"]
            views = list(base_views) + [nested["scores"][arm][fold_id]]
            weights, scales, info, _normalized = fit_fusion(
                config, views, hard, table, eligible, complement,
            )
            model = fit_centroids(arm, train.emb, labels, parses, hard, complement)
            held_geometry = score_centroids(model, train.emb[held])
            held_views = [
                oof["full_mean_a"][held], oof["full_mean_b"][held], oof["full_tree"][held], held_geometry,
            ]
            held_bank = slice_bank(train.bank, held)
            choices, scored = decode_held(
                config, weights, scales, held_views, labels[held], hard[held], held_bank,
            )
            reference_choices = replayed[fold_id]["choices"]
            reference_counts = replayed[fold_id]["counts"]
            alone_scale = None if scales is None else [scales[-1]]
            geometry_choices, geometry_score = decode_held(
                config, [1.0], alone_scale, [held_geometry], labels[held], hard[held], held_bank,
            )
            arm_scale = None if record["scales"] is None else [record["scales"][0]]
            arm_choices, arm_score = decode_held(
                config, [1.0], arm_scale, [oof["full_mean_a"][held]], labels[held], hard[held], held_bank,
            )
            fusion_hits = hits_of(choices, labels[held])
            reference_hits = hits_of(reference_choices, labels[held])
            geometry_hits = hits_of(geometry_choices, labels[held])
            arm_hits = hits_of(arm_choices, labels[held])
            versus_reference = flip_counts(fusion_hits, reference_hits)
            versus_arm = flip_counts(geometry_hits, arm_hits)
            for key in pooled_flips:
                pooled_flips[key] += versus_reference[key]
                pooled_arm[key] += versus_arm[key]
            held_hard = hard[held]
            hard_candidates = np.repeat(held_hard[:, None], 4, axis=1)
            held_eligible, _held_parses = question_eligibility(held_bank)
            score_corr = correlation(
                oof["full_mean_a"][held], held_geometry, hard_candidates & held_eligible,
            )
            correlations.append(score_corr)
            reference_folds.append({
                "hard_correct": int(reference_counts["hard_correct"]),
                "hard_total": int(reference_counts["hard_total"]),
                "all_correct": int(reference_counts["all_correct"]),
                "all_total": int(reference_counts["all_total"]),
            })
            selected_folds.append({
                "hard_correct": int(scored["hard_correct"]),
                "hard_total": int(scored["hard_total"]),
                "all_correct": int(scored["all_correct"]),
                "all_total": int(scored["all_total"]),
            })
            fold_records.append({
                "outer_fold": fold_id,
                "decoder": config,
                "selected_arm": arm,
                "selection": selection["pooled"],
                "standalone_inner": selection["standalone"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "fit_loss": info["loss"],
                "reference": reference_folds[-1],
                "selected": selected_folds[-1],
                "standalone_outer": {
                    "hard_correct": int(geometry_score["hard_correct"]),
                    "hard_total": int(geometry_score["hard_total"]),
                    "all_correct": int(geometry_score["all_correct"]),
                    "all_total": int(geometry_score["all_total"]),
                },
                "arm_a_standalone_outer": {
                    "hard_correct": int(arm_score["hard_correct"]),
                    "hard_total": int(arm_score["hard_total"]),
                    "all_correct": int(arm_score["all_correct"]),
                    "all_total": int(arm_score["all_total"]),
                },
                "flips_vs_reference": versus_reference,
                "geometry_vs_arm_a": versus_arm,
                "score_correlation_with_arm_a": score_corr,
                "class_counts": model["class_counts"],
                "requested_k": model["requested_k"],
                "actual_k": model["actual_k"],
                "tau": model["tau"],
                "centroid_fit_seconds": model["fit_seconds"],
            })
            print(
                f"  outer {fold_id}: {arm} hard {scored['hard_correct']}/{scored['hard_total']} "
                f"vs lock {reference_counts['hard_correct']}/{reference_counts['hard_total']}",
                flush=True,
            )

        gate = internal_gate_decision(reference_folds, selected_folds, 0.5, 3)
        gate["minimum_mean_fold_hard_gain_percentage_points"] = 0.5
        gate["minimum_pooled_hard_gain_percentage_points"] = 0.5
        gate["minimum_positive_outer_folds"] = 3
        gate["validation_opened"] = False
        print(
            f"Gate mean {gate['mean_hard_gain_percentage_points']:.4f} pp, "
            f"pooled {gate['pooled_hard_gain_percentage_points']:.4f} pp, "
            f"positive folds {gate['positive_outer_folds']}/5, "
            f"all delta {gate['pooled_all_delta']}, passed={gate['passed']}",
            flush=True,
        )
        finite_correlations = [value for value in correlations if value is not None]
        audit.update({
            "environment": environment,
            "source_drift": drifted,
            "b25_oof_sha256": sha256_file(B25_DIR / "oof_logits.npz"),
            "training": {
                "nested_centroid_fits": 40,
                "outer_centroid_fits": 5,
                "nested_fit_seconds": nested["seconds"],
                "new_attention_models": 0,
                "kmeans_random_state": 42,
            },
            "outer_folds": fold_records,
            "diagnostics": {
                "fusion_vs_locked_three_view": pooled_flips,
                "standalone_geometry_vs_arm_a": pooled_arm,
                "mean_score_correlation_with_arm_a": (
                    None if not finite_correlations else float(np.mean(finite_correlations))
                ),
            },
            "internal_gate": gate,
        })
        if not gate["passed"]:
            audit["verdict"] = "STOPPED_INTERNAL_GATE"
            print("Internal gate did not pass. Validation stays closed.", flush=True)
        else:
            print("Selecting the full-train centroid arm", flush=True)
            full_geometry = {}
            full_seconds = {}
            for arm in ARMS:
                full_geometry[arm], full_seconds[arm] = score_holdout_geometry(
                    train.emb, labels, parses, hard, folds, arm,
                )
            final = select_arm(
                full_lock["config"],
                [oof["full_mean_a"], oof["full_mean_b"], oof["full_tree"]],
                full_geometry, labels, hard, train.bank, parses, eligible, table, folds, [],
            )
            chosen = final["arm"]
            all_rows = np.arange(len(train.bank))
            fused_views = [oof["full_mean_a"], oof["full_mean_b"], oof["full_tree"], full_geometry[chosen]]
            weights, scales, info, _normalized = fit_fusion(
                full_lock["config"], fused_views, hard, table, eligible, all_rows,
            )
            model = fit_centroids(chosen, train.emb, labels, parses, hard, all_rows)
            centroid_path = EXP_DIR / "centroid_model.npz"
            np.savez(
                centroid_path,
                correct=np.asarray(model["correct"]),
                wrong=np.asarray(model["wrong"]),
                tau=np.asarray([] if model["tau"] is None else [model["tau"]], dtype=np.float64),
                arm=np.asarray(chosen),
            )
            centroid_sha = sha256_file(centroid_path)
            candidate = {
                "experiment": "EXP-B26",
                "arm": chosen,
                "config": full_lock["config"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "weight_fit": info,
                "selection": final["pooled"],
                "standalone_inner": final["standalone"],
                "class_counts": model["class_counts"],
                "requested_k": model["requested_k"],
                "actual_k": model["actual_k"],
                "tau": model["tau"],
                "centroid_sha256": centroid_sha,
                "fit_scope": "full-train inner OOF for the arm choice; centroids refit on all train-hard rows",
            }
            lock_sha = write_json(EXP_DIR / "downstream_centroid_lock.json", candidate)
            register_seeds("EXP-B26", [{
                "arm": chosen,
                "seed": 42,
                "member": "centroid",
                "derived": {"kmeans_random_state": 42, "n_init": 10, "max_iter": 300},
                "file": rel(centroid_path),
                "sha256": centroid_sha,
            }])
            audit["full_train_selection"] = {
                "arm": chosen,
                "selection": final["pooled"],
                "weights": candidate["weights"],
                "scales": scales,
                "tau": model["tau"],
                "actual_k": model["actual_k"],
                "holdout_fit_seconds": full_seconds,
                "lock_sha256": lock_sha,
                "centroid_sha256": centroid_sha,
            }
            audit["validation_opened"] = True
            gate["validation_opened"] = True
            confirmation = run_validation_confirmation(
                {**candidate, "model": model},
                {
                    "labels": labels,
                    "hard": hard,
                    "table": table,
                    "eligible": eligible,
                    "mean_a": oof["full_mean_a"],
                    "mean_b": oof["full_mean_b"],
                    "tree": oof["full_tree"],
                    "geometry_oof": full_geometry[chosen],
                    "per_seed_a": {int(seed): oof[f"full_a_seed_{seed}"] for seed in SEEDS},
                    "per_seed_b": {int(seed): oof[f"full_b_seed_{seed}"] for seed in SEEDS},
                },
                contract,
            )
            audit["validation_confirmation"] = confirmation
            audit["validation_queries"] = 1
            audit["verdict"] = confirmation["retention_gate"]["verdict"]
            print(f"Validation verdict: {audit['verdict']}", flush=True)
    except Exception:
        audit["error"] = traceback.format_exc()
        print(audit["error"], flush=True)
        if audit["verdict"] not in {"STOPPED_INTERNAL_GATE", "RETAINED", "VALIDATION_NOT_RETAINED"}:
            audit["verdict"] = "FAILED"
        raise
    finally:
        audit["execution_seconds"] = time.perf_counter() - started
        audit["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_json(AUDIT_PATH, audit)
        print(f"VERDICT: {audit['verdict']}", flush=True)
        print(f"Audit written to {AUDIT_PATH}", flush=True)
    return audit


if __name__ == "__main__":
    run_exp_b26()
