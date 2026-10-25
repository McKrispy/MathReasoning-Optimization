"""EXP-B27: add one Bradley-Terry energy view to the B25 locked three-view decoder.

E0 is the existing Arm A listwise model and is reused from the B25 cache.
E1 and E2 keep that architecture and the B12 schedule. Each outer fold chooses
E1 or E2 on inner OOF, then the outer held-out questions score that choice.
Diagnostic temperatures do not enter the fusion. Validation stays closed when
the fused system misses the internal gate.
"""
from __future__ import annotations

import hashlib
import math
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b07_cross_attn import HPARAMS, kfold_indices  # noqa: E402
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
    retention_decision,
    write_json,
)
from experiments.exp_b26_centroid_verifier import (  # noqa: E402
    check_sources,
    correlation,
    decode_held,
    fit_fusion,
    flip_counts,
    hits_of,
    load_b25_caches,
    replay_locked_reference,
    slice_bank,
)
from experiments.multiseed import (  # noqa: E402
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    configure_determinism,
    environment_record,
    fold_model_seeds,
    full_model_seeds,
    register_seeds,
    rel,
)
from models.cross_attn_net import CrossCandidateAttentionNet, uncompressed_weight_bytes  # noqa: E402
from models.energy_verifier import (  # noqa: E402
    DIAGNOSTIC_TEMPERATURES,
    SELECTABLE_ARMS,
    VARIANTS,
    fit_energy_model,
    select_diagnostic_temperature,
    standalone_report,
)
from postprocessing.aligned_fusion import (  # noqa: E402
    build_cluster_table,
    internal_gate_decision,
    nested_attention_seeds,
    normalize_views,
    question_eligibility,
    score_aligned,
    select_config,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
N_FOLDS = 5
EXP_DIR = RUNS / "exp_b27"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b27_energy_audit.json"
CONTRACT_PATH = RUNS / "audit_logs" / "private_configuration_omitted.json"
B25_DIR = RUNS / "exp_b25"


def blank_training() -> Dict[str, float]:
    return {
        "models": 0,
        "fit_seconds": 0.0,
        "warmup_updates": 0,
        "finetune_updates": 0,
        "skipped_batches": 0,
        "all_positive_batches": 0,
        "mixed_groups": 0,
        "pairs": 0,
        "warmup_mixed_questions": 0,
        "finetune_mixed_questions": 0,
        "warmup_pairs": 0,
        "finetune_pairs": 0,
    }


def add_training(bucket: Dict[str, float], stats: Dict[str, Any]) -> None:
    bucket["models"] += 1
    bucket["fit_seconds"] += float(stats["fit_seconds"])
    bucket["warmup_updates"] += int(stats["warmup"]["optimizer_steps"])
    bucket["finetune_updates"] += int(stats["finetune"]["optimizer_steps"])
    for phase in ("warmup", "finetune"):
        bucket["skipped_batches"] += int(stats[phase]["skipped_batches"])
        bucket["all_positive_batches"] += int(stats[phase]["all_positive_batches"])
        bucket["mixed_groups"] += int(stats[phase]["mixed_groups"])
        bucket["pairs"] += int(stats[phase]["pairs"])
        bucket[f"{phase}_mixed_questions"] += int(stats["fit_groups"][phase]["mixed_questions"])
        bucket[f"{phase}_pairs"] += int(stats["fit_groups"][phase]["pairs"])


def assert_hparams(batch: Dict[str, Any]) -> None:
    for key, expected in batch["hparams_snapshot"].items():
        actual = HPARAMS[key]
        if isinstance(expected, float) or isinstance(actual, float):
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"HPARAMS[{key}] is {actual}, contract snapshot says {expected}")
        elif actual != expected:
            raise RuntimeError(f"HPARAMS[{key}] is {actual}, contract snapshot says {expected}")


def assert_energy_seeds() -> None:
    seen = []
    for variant in (0, 1, 2, 3):
        for seed in SEEDS:
            for outer in range(N_FOLDS):
                for inner in range(N_FOLDS):
                    derived = nested_attention_seeds(variant, seed, outer, inner)
                    seen.extend(derived.values())
    for seed in SEEDS:
        seen.extend(full_model_seeds(int(seed)).values())
        for fold_id in range(N_FOLDS):
            seen.extend(fold_model_seeds(int(seed), fold_id).values())
    if len(seen) != len(set(seen)):
        raise RuntimeError("energy seed streams collide with each other or with B12/B25 seeds")


def train_mean(
    embeddings: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    fit_rows: np.ndarray,
    held: np.ndarray,
    outer: int,
    inner: int,
    device,
    stats: Dict[str, Dict[str, float]],
    keep_seeds: bool,
) -> Dict[str, Any]:
    fit_hard = fit_rows[np.asarray(hard, dtype=bool)[fit_rows]]
    if fit_hard.size == 0:
        raise RuntimeError(f"energy fit outer {outer} inner {inner} has no hard questions")
    means = {}
    per_seed: Dict[str, Dict[int, np.ndarray]] = {}
    for arm in SELECTABLE_ARMS:
        total = np.zeros((held.size, 4), dtype=np.float64)
        if keep_seeds:
            per_seed[arm] = {}
        for seed in SEEDS:
            derived = nested_attention_seeds(VARIANTS[arm], int(seed), outer, inner)
            state, one = fit_energy_model(
                embeddings[fit_rows], labels[fit_rows],
                embeddings[fit_hard], labels[fit_hard],
                derived, device, arm,
            )
            logits = CrossCandidateAttentionNet(state).forward_numpy(embeddings[held])
            total += np.asarray(logits, dtype=np.float64)
            if keep_seeds:
                per_seed[arm][int(seed)] = np.asarray(logits, dtype=np.float32)
            add_training(stats[arm], one)
        means[arm] = (total / float(len(SEEDS))).astype(np.float32)
    return {"mean": means, "per_seed": per_seed}


def select_energy_arm(
    config: Dict[str, Any],
    base_views: Sequence[np.ndarray],
    energy: Dict[str, np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank,
    parses,
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    folds: Sequence[np.ndarray],
    excluded: Sequence[int],
) -> Dict[str, Any]:
    pooled = []
    standalone = []
    banned = {int(fold_id) for fold_id in excluded}
    inners = [fold_id for fold_id in range(N_FOLDS) if fold_id not in banned]
    for arm in SELECTABLE_ARMS:
        views = list(base_views) + [energy[arm]]
        hard_correct = all_correct = hard_total = all_total = absent = 0
        standalone_hard = standalone_all = 0
        nll_sum = 0.0
        for inner in inners:
            fit_rows = complement_rows(folds, list(banned | {inner}))
            eval_rows = folds[inner]
            weights, _scales, _info, normalized = fit_fusion(
                config, views, hard, table, eligible, fit_rows,
            )
            scored = score_aligned(
                normalized, labels, hard, bank, parses, eligible, table,
                weights, float(config["temperature"]), float(config["rho"]), eval_rows,
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
    return {"arm": SELECTABLE_ARMS[index], "index": index, "pooled": pooled, "standalone": standalone}


def diagnostic_for_rows(
    scores: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    bank,
    parses,
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
) -> Dict[str, Any]:
    """Choose a standalone temperature on fit-partition OOF, then score the held rows."""
    grid = [
        standalone_report(
            scores, labels, hard, bank, parses, eligible, table, fit_rows, temperature,
        )
        for temperature in DIAGNOSTIC_TEMPERATURES
    ]
    chosen = select_diagnostic_temperature(grid)
    temperature = float(grid[chosen]["temperature"])
    held = standalone_report(
        scores, labels, hard, bank, parses, eligible, table, held_rows, temperature,
    )
    return {
        "chosen_temperature": temperature,
        "fit": grid[chosen],
        "held": held,
        "fit_nll_by_temperature": [
            {"temperature": float(item["temperature"]), "nll_mean": float(item["nll_mean"])}
            for item in grid
        ],
    }


def argmax_hard(scores: np.ndarray, labels: np.ndarray, hard: np.ndarray) -> int:
    choice = np.argmax(np.asarray(scores), axis=1)
    hits = labels[np.arange(labels.shape[0]), choice] >= 0.5
    return int(hits[np.asarray(hard, dtype=bool)].sum())


def run_validation_confirmation(
    candidate: Dict[str, Any],
    train_pack: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """One frozen four-view candidate. Energy weights are already fit on train only."""
    val = SplitData("validation")
    val_hard = np.asarray(val.hard, dtype=bool)
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    tree_val = np.asarray(np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")["seed_42"])
    per_seed_energy = {
        int(seed): CrossCandidateAttentionNet(state).forward_numpy(val.emb)
        for seed, state in candidate["states"].items()
    }
    energy_val = np.mean(list(per_seed_energy.values()), axis=0).astype(np.float32)
    ensemble_views = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], tree_val, energy_val]
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
    paired = []
    for seed in SEEDS:
        seed_views = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], tree_val]
        old_weights, old_scales, _old_info, _old_norm = fit_fusion(
            config,
            [train_pack["per_seed_a"][int(seed)], train_pack["per_seed_b"][int(seed)], train_pack["tree"]],
            hard, table, eligible, np.arange(labels.shape[0]),
        )
        _old_choice, old_score = decode_held(
            config, old_weights, old_scales, seed_views, val.labels, val_hard, val.bank,
        )
        new_weights, new_scales, _new_info, _new_norm = fit_fusion(
            config,
            [
                train_pack["per_seed_a"][int(seed)],
                train_pack["per_seed_b"][int(seed)],
                train_pack["tree"],
                train_pack["per_seed_energy"][int(seed)],
            ],
            hard, table, eligible, np.arange(labels.shape[0]),
        )
        _new_choice, new_score = decode_held(
            config, new_weights, new_scales,
            seed_views + [per_seed_energy[int(seed)]], val.labels, val_hard, val.bank,
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
    prediction_path = EXP_DIR / "exp_b27_validation_predictions.json"
    score_path = EXP_DIR / "exp_b27_validation_score.json"
    write_official_predictions(prediction_path, val.bank, ensemble_choice)
    official = official_score(prediction_path, score_path)
    if int(official["correct"]) != int(ensemble_score["all_correct"]):
        raise RuntimeError("official all-count does not match the in-process four-view decode")
    new_hits = hits_of(ensemble_choice, val.labels)
    old_hits = hits_of(reference_choice, val.labels)
    asset_bytes = int(contract["baseline"]["historical"]["recorded_uncompressed_bytes"])
    lock_bytes = (B25_DIR / "downstream_decoder_lock.json").stat().st_size
    energy_bytes = int(sum(path.stat().st_size for path in candidate["weight_paths"]))
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
        "same_protocol_reference": {
            "hard_correct": int(hits_of(reference_choice, val.labels)[val_hard].sum()),
            "hard_total": int(val_hard.sum()),
            "all_correct": int(hits_of(reference_choice, val.labels).sum()),
            "all_total": int(len(val.bank)),
        },
        "paired": paired,
        "flips_vs_locked_three_view": flip_counts(new_hits, old_hits),
        "official_score": official,
        "in_process_decode_seconds": decode_seconds,
        "asset_preaudit": {
            "scope": "local in-process decode plus unchanged B20 assets, the B25 lock, and the energy weights",
            "not_a_fresh_predict_process": True,
            "not_a_xeon_certificate": True,
            "b20_uncompressed_bytes": asset_bytes,
            "b25_lock_bytes": int(lock_bytes),
            "energy_bytes": energy_bytes,
            "combined_bytes": int(asset_bytes + lock_bytes + energy_bytes),
        },
        "validation_configurations": 4,
        "selected_arm": candidate["arm"],
    }
    confirmation["retention_gate"] = retention_decision(confirmation, contract)
    return confirmation


def run_exp_b27() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B27: energy view on the locked decoder")
    print("=" * 78, flush=True)
    started = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "experiment": "EXP-B27",
        "protocol": PROTOCOL,
        "reference_system": "B25 fold-local locked three-view decoder",
        "e0_reused_from_b25": True,
        "validation_queries": 0,
        "validation_opened": False,
        "test_references_opened": False,
        "written_to_model_bundle": False,
        "candidate_submission_updated": False,
        "validation_confirmation": None,
        "verdict": "FAILED",
    }
    try:
        contract = json_load(CONTRACT_PATH)
        batch = contract["batches"]["EXP-B27"]
        if int(batch["max_new_energy_loss_configurations"]) != 2:
            raise RuntimeError("B27 new-loss budget is not 2")
        if tuple(batch["arms"]) != ("E0", "E1", "E2"):
            raise RuntimeError("B27 arm order is not E0, E1, E2")
        grid = [float(item) for item in contract["calibration_metrics"]["diagnostic_temperature_grid"]]
        if [float(item) for item in DIAGNOSTIC_TEMPERATURES] != grid:
            raise RuntimeError("diagnostic temperature grid drifted")
        assert_hparams(batch)
        assert_energy_seeds()
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

        stats = {arm: blank_training() for arm in SELECTABLE_ARMS}
        nested = {arm: np.zeros((N_FOLDS, len(train.bank), 4), dtype=np.float32) for arm in SELECTABLE_ARMS}
        filled = np.zeros((N_FOLDS, len(train.bank)), dtype=bool)
        cell_id = 0
        for outer in range(N_FOLDS):
            for inner in range(N_FOLDS):
                if inner == outer:
                    continue
                cell_id += 1
                fit_rows = complement_rows(folds, [outer, inner])
                held = folds[inner]
                if set(int(row) for row in fit_rows).intersection(int(row) for row in held):
                    raise RuntimeError(f"energy cell outer {outer} inner {inner} leaked")
                print(f"  nested energy {cell_id}/20 outer {outer} inner {inner}", flush=True)
                trained = train_mean(
                    train.emb, labels, hard, fit_rows, held, outer, inner, device, stats, False,
                )
                for arm in SELECTABLE_ARMS:
                    nested[arm][outer, held] = trained["mean"][arm]
                filled[outer, held] = True
                print(
                    f"    cumulative E1 {stats['E1']['fit_seconds']:.1f}s "
                    f"E2 {stats['E2']['fit_seconds']:.1f}s",
                    flush=True,
                )

        print("Training exclude-one-fold E1/E2 models", flush=True)
        outer_mean = {arm: np.zeros((len(train.bank), 4), dtype=np.float32) for arm in SELECTABLE_ARMS}
        outer_seed = {
            arm: {int(seed): np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
            for arm in SELECTABLE_ARMS
        }
        for fold_id, held in enumerate(folds):
            fit_rows = complement_rows(folds, [fold_id])
            print(f"  outer energy fold {fold_id}", flush=True)
            trained = train_mean(
                train.emb, labels, hard, fit_rows, held, fold_id, fold_id, device, stats, True,
            )
            for arm in SELECTABLE_ARMS:
                outer_mean[arm][held] = trained["mean"][arm]
                for seed in SEEDS:
                    outer_seed[arm][int(seed)][held] = trained["per_seed"][arm][int(seed)]

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
            if not bool(filled[fold_id, complement].all()):
                raise RuntimeError(f"outer {fold_id} is missing nested energy scores")
            config = record["selected_config"]
            base_views = [
                oof["nested_mean_a"][fold_id],
                oof["nested_mean_b"][fold_id],
                oof["nested_tree"][fold_id],
            ]
            print(f"Selecting energy arm on outer {fold_id}", flush=True)
            selection = select_energy_arm(
                config, base_views,
                {arm: nested[arm][fold_id] for arm in SELECTABLE_ARMS},
                labels, hard, train.bank, parses, eligible, table, folds, [fold_id],
            )
            arm = selection["arm"]
            views = list(base_views) + [nested[arm][fold_id]]
            weights, scales, info, _normalized = fit_fusion(
                config, views, hard, table, eligible, complement,
            )
            held_bank = slice_bank(train.bank, held)
            held_views = [
                oof["full_mean_a"][held],
                oof["full_mean_b"][held],
                oof["full_tree"][held],
                outer_mean[arm][held],
            ]
            choices, scored = decode_held(
                config, weights, scales, held_views, labels[held], hard[held], held_bank,
            )
            reference_counts = replayed[fold_id]["counts"]
            reference_choices = replayed[fold_id]["choices"]
            alone_scale = None if scales is None else [scales[-1]]
            energy_choices, energy_score = decode_held(
                config, [1.0], alone_scale, [outer_mean[arm][held]], labels[held], hard[held], held_bank,
            )
            arm_scale = None if record["scales"] is None else [record["scales"][0]]
            arm_choices, arm_score = decode_held(
                config, [1.0], arm_scale, [oof["full_mean_a"][held]], labels[held], hard[held], held_bank,
            )
            other = "E1" if arm == "E2" else "E2"
            _other_norm, other_scales = normalize_views(
                [nested[other][fold_id]], config["normalization"], eligible, rows=complement,
            )
            _other_choices, other_score = decode_held(
                config, [1.0], other_scales, [outer_mean[other][held]], labels[held], hard[held], held_bank,
            )
            fusion_hits = hits_of(choices, labels[held])
            reference_hits = hits_of(reference_choices, labels[held])
            energy_hits = hits_of(energy_choices, labels[held])
            arm_hits = hits_of(arm_choices, labels[held])
            versus_reference = flip_counts(fusion_hits, reference_hits)
            versus_arm = flip_counts(energy_hits, arm_hits)
            for key in pooled_flips:
                pooled_flips[key] += versus_reference[key]
                pooled_arm[key] += versus_arm[key]
            held_hard = hard[held]
            hard_candidates = np.repeat(held_hard[:, None], 4, axis=1)
            held_eligible, _held_parses = question_eligibility(held_bank)
            score_corr = correlation(
                oof["full_mean_a"][held], outer_mean[arm][held], hard_candidates & held_eligible,
            )
            correlations.append(score_corr)
            seed_hard = [
                argmax_hard(outer_seed[arm][int(seed)][held], labels[held], hard[held])
                for seed in SEEDS
            ]
            calibration = {
                "E0": diagnostic_for_rows(
                    _full_with_nested(oof["nested_mean_a"][fold_id], oof["full_mean_a"], held),
                    labels, hard, train.bank, parses, eligible, table, complement, held,
                ),
            }
            for energy_arm in SELECTABLE_ARMS:
                view = nested[energy_arm][fold_id].copy()
                view[held] = outer_mean[energy_arm][held]
                calibration[energy_arm] = diagnostic_for_rows(
                    view, labels, hard, train.bank, parses, eligible, table, complement, held,
                )
            reference_row = {
                "hard_correct": int(reference_counts["hard_correct"]),
                "hard_total": int(reference_counts["hard_total"]),
                "all_correct": int(reference_counts["all_correct"]),
                "all_total": int(reference_counts["all_total"]),
            }
            selected_row = {
                "hard_correct": int(scored["hard_correct"]),
                "hard_total": int(scored["hard_total"]),
                "all_correct": int(scored["all_correct"]),
                "all_total": int(scored["all_total"]),
            }
            reference_folds.append(reference_row)
            selected_folds.append(selected_row)
            fold_records.append({
                "outer_fold": fold_id,
                "decoder": config,
                "selected_arm": arm,
                "selection": selection["pooled"],
                "standalone_inner": selection["standalone"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "fit_loss": info["loss"],
                "reference": reference_row,
                "selected": selected_row,
                "standalone_outer": {
                    "hard_correct": int(energy_score["hard_correct"]),
                    "hard_total": int(energy_score["hard_total"]),
                    "all_correct": int(energy_score["all_correct"]),
                    "all_total": int(energy_score["all_total"]),
                },
                "other_arm_standalone_outer": {
                    "arm": other,
                    "hard_correct": int(other_score["hard_correct"]),
                    "hard_total": int(other_score["hard_total"]),
                    "all_correct": int(other_score["all_correct"]),
                    "all_total": int(other_score["all_total"]),
                },
                "arm_a_standalone_outer": {
                    "hard_correct": int(arm_score["hard_correct"]),
                    "hard_total": int(arm_score["hard_total"]),
                    "all_correct": int(arm_score["all_correct"]),
                    "all_total": int(arm_score["all_total"]),
                },
                "per_seed_argmax_hard": seed_hard,
                "argmax_hard_mean": float(np.mean(seed_hard)),
                "argmax_hard_sample_std": float(np.std(seed_hard, ddof=1)),
                "flips_vs_reference": versus_reference,
                "energy_vs_arm_a": versus_arm,
                "score_correlation_with_arm_a": score_corr,
                "calibration": calibration,
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
            "b25_oof_sha256": load_b25_sha(),
            "training": {
                "arms": stats,
                "nested_models": 120,
                "outer_models": 30,
                "e0_models": 0,
                "new_attention_recipe": False,
            },
            "outer_folds": fold_records,
            "diagnostics": {
                "fusion_vs_locked_three_view": pooled_flips,
                "standalone_energy_vs_arm_a": pooled_arm,
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
            print("Selecting the full-train energy arm", flush=True)
            final = select_energy_arm(
                full_lock["config"],
                [oof["full_mean_a"], oof["full_mean_b"], oof["full_tree"]],
                outer_mean, labels, hard, train.bank, parses, eligible, table, folds, [],
            )
            chosen = final["arm"]
            all_rows = np.arange(len(train.bank))
            fused_views = [oof["full_mean_a"], oof["full_mean_b"], oof["full_tree"], outer_mean[chosen]]
            weights, scales, info, _normalized = fit_fusion(
                full_lock["config"], fused_views, hard, table, eligible, all_rows,
            )
            states = {}
            weight_paths = []
            full_stats = blank_training()
            fit_hard = all_rows[hard]
            for seed in SEEDS:
                state, one = fit_energy_model(
                    train.emb, labels, train.emb[fit_hard], labels[fit_hard],
                    full_model_seeds(int(seed)), device, chosen,
                )
                path = EXP_DIR / f"energy_{chosen}_seed_{seed}.npz"
                np.savez(path, **state)
                states[int(seed)] = state
                weight_paths.append(path)
                add_training(full_stats, one)
            payload = {
                "experiment": "EXP-B27",
                "arm": chosen,
                "config": full_lock["config"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "weight_fit": info,
                "selection": final["pooled"],
                "files": [rel(path) for path in weight_paths],
                "uncompressed_weight_bytes": int(sum(uncompressed_weight_bytes(state) for state in states.values())),
                "fit_scope": "full-train inner OOF for the arm choice; energy refit on all train rows",
            }
            lock_sha = write_json(EXP_DIR / "downstream_energy_lock.json", payload)
            register_seeds("EXP-B27", [
                {
                    "arm": chosen,
                    "seed": int(seed),
                    "member": "energy",
                    "derived": full_model_seeds(int(seed)),
                    "file": rel(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for seed, path in zip(SEEDS, weight_paths)
            ])
            audit["full_train_selection"] = {
                "arm": chosen,
                "selection": final["pooled"],
                "weights": payload["weights"],
                "scales": scales,
                "full_train_fit": full_stats,
                "lock_sha256": lock_sha,
            }
            audit["validation_opened"] = True
            gate["validation_opened"] = True
            confirmation = run_validation_confirmation(
                {**payload, "states": states, "weight_paths": weight_paths, "model": None},
                {
                    "labels": labels,
                    "hard": hard,
                    "table": table,
                    "eligible": eligible,
                    "mean_a": oof["full_mean_a"],
                    "mean_b": oof["full_mean_b"],
                    "tree": oof["full_tree"],
                    "per_seed_energy": outer_seed[chosen],
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


def json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_b25_sha() -> str:
    from experiments.exp_b25_decoder_alignment import sha256_file
    return sha256_file(B25_DIR / "oof_logits.npz")


def _full_with_nested(nested_scores: np.ndarray, full_scores: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Fit rows keep nested E0 scores. Held rows carry the exclude-one-fold E0 scores."""
    view = np.array(nested_scores, dtype=np.float64, copy=True)
    view[held] = np.asarray(full_scores, dtype=np.float64)[held]
    return view


if __name__ == "__main__":
    run_exp_b27()
