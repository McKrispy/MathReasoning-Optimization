"""EXP-B25: fit fusion weights with the same mask, temperature and clusters used at decode.

The base models stay on the B12 recipe. Nested retraining exists only so each
outer fold can choose a decoder without seeing its held-out questions. Seven
configurations are scored. The internal gate requires both hard-accuracy
definitions to gain at least 0.5 percentage points, at least three outer folds
to improve, and the pooled all-correct count not to fall. Validation stays
closed when that gate fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b07_cross_attn import HPARAMS, kfold_indices, standardize_apply, standardize_fit  # noqa: E402
from experiments.exp_b15_tree import fit_tree, tree_logits  # noqa: E402
from experiments.exp_b22_g05_r5 import (  # noqa: E402
    B20_STRICT_ALL,
    B20_STRICT_HARD,
    B20_STRICT_WEIGHTS,
    HISTORICAL_ALL,
    HISTORICAL_HARD,
    HISTORICAL_WEIGHTS,
    fold_svd_blocks,
    fit_three_weights,
    official_score,
    score_choice,
    system_choice,
    write_official_predictions,
)
from experiments.exp_b23_hard_weighting import complement_rows  # noqa: E402
from experiments.multiseed import (  # noqa: E402
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    configure_determinism,
    environment_record,
    fold_model_seeds,
    reasoning_struct,
    register_seeds,
    rel,
    train_model,
)
from models.cross_attn_net import CrossCandidateAttentionNet  # noqa: E402
from postprocessing.aligned_fusion import (  # noqa: E402
    b25_configs,
    build_cluster_table,
    fit_aligned_weights,
    internal_gate_decision,
    nested_attention_seeds,
    normalize_views,
    question_eligibility,
    score_aligned,
    select_config,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
N_FOLDS = 5
PUBLISHED_ARM_A_HARD = ((345, 444), (330, 424), (289, 392), (325, 416), (351, 438))
STRUCT_ATOL = 1e-4

EXP_DIR = RUNS / "exp_b25"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b25_decoder_alignment_audit.json"
CONTRACT_PATH = RUNS / "audit_logs" / "private_configuration_omitted.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def verify_sources(contract: Dict[str, Any]) -> None:
    mismatches = []
    for relative, expected in contract["source_sha256"].items():
        path = WORKSPACE / relative
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append({"file": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"source hash mismatch before B25: {mismatches}")


def load_contract() -> Dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    gate = contract["nested_protocol"]["new_internal_gate"]
    if float(gate["minimum_mean_fold_hard_gain_percentage_points"]) != 0.5:
        raise RuntimeError("B25 mean-fold gate is not 0.5 percentage points")
    if float(gate["minimum_pooled_hard_gain_percentage_points"]) != 0.5:
        raise RuntimeError("B25 pooled gate is not 0.5 percentage points")
    if int(gate["minimum_positive_outer_folds"]) != 3:
        raise RuntimeError("B25 positive-fold gate is not 3")
    if int(gate["minimum_pooled_all_correct_delta"]) != 0:
        raise RuntimeError("B25 pooled all-correct delta gate is not 0")
    if contract["batches"]["EXP-B25"]["new_base_training_recipe"] is not False:
        raise RuntimeError("B25 must not change the base training recipe")
    return contract


def count_choice(choice: Sequence[int], labels: np.ndarray, hard: np.ndarray) -> Dict[str, int]:
    chosen = np.asarray(choice, dtype=np.int64)
    hits = labels[np.arange(labels.shape[0]), chosen] >= 0.5
    hard = np.asarray(hard, dtype=bool)
    return {
        "hard_correct": int(hits[hard].sum()),
        "hard_total": int(hard.sum()),
        "all_correct": int(hits.sum()),
        "all_total": int(hits.size),
    }


def slice_bank(bank: Sequence[Dict[str, Any]], rows: np.ndarray) -> List[Dict[str, Any]]:
    return [bank[int(row)] for row in rows]


def train_attention(
    embeddings: np.ndarray,
    struct: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
    seeds: Dict[str, int],
    device,
    arm: str,
) -> Tuple[np.ndarray, int, float]:
    """B12 recipe: warmup uses every fit row, finetune uses the hard fit rows."""
    fit_hard = fit_rows[np.asarray(hard, dtype=bool)[fit_rows]]
    if fit_hard.size == 0:
        raise RuntimeError(f"{arm} fit has no hard questions")
    if arm == "arm_a":
        features = embeddings
    elif arm == "arm_b":
        mean, std = standardize_fit(struct[fit_rows])
        features = np.concatenate(
            [embeddings, standardize_apply(struct, mean, std)], axis=-1,
        ).astype(np.float32)
    else:
        raise ValueError(arm)
    state, stats = train_model(
        features[fit_rows],
        labels[fit_rows],
        features[fit_hard],
        labels[fit_hard],
        seeds,
        device,
    )
    logits = CrossCandidateAttentionNet(state).forward_numpy(features[held_rows])
    steps = int(stats["warmup"]["optimizer_steps"]) + int(stats["finetune"]["optimizer_steps"])
    return np.asarray(logits, dtype=np.float32), steps, float(stats["fit_seconds"])


def train_tree(
    embeddings: np.ndarray,
    struct: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
) -> Tuple[np.ndarray, float]:
    svd_fit, svd_held = fold_svd_blocks(embeddings[fit_rows], embeddings[held_rows])
    fit_features = np.concatenate([struct[fit_rows], svd_fit], axis=-1)
    held_features = np.concatenate([struct[held_rows], svd_held], axis=-1)
    model, seconds = fit_tree("M04a", fit_features, labels[fit_rows], 42)
    return np.asarray(tree_logits("M04a", model, held_features), dtype=np.float32), float(seconds)


def verify_saved_folds(
    struct: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Reuse B12 only when Arm B normalization and the published Arm A counts match."""
    gaps = []
    reusable = True
    for seed in SEEDS:
        for fold_id, held in enumerate(folds):
            fit_rows = complement_rows(folds, [fold_id])
            mean, std = standardize_fit(struct[fit_rows])
            path = RUNS / f"exp_b12/seed_{seed}/arm_embeddings_plus_g01_g03_fold{fold_id}.npz"
            arm_a_path = RUNS / f"exp_b12/seed_{seed}/arm_embeddings_only_fold{fold_id}.npz"
            if not path.exists() or not arm_a_path.exists():
                reusable = False
                gaps.append({"seed": int(seed), "fold": fold_id, "missing": True})
                continue
            saved = np.load(path)
            gap_mean = float(np.max(np.abs(np.asarray(saved["struct_mean"]) - mean)))
            gap_std = float(np.max(np.abs(np.asarray(saved["struct_std"]) - std)))
            gaps.append({
                "seed": int(seed),
                "fold": fold_id,
                "max_abs_mean": gap_mean,
                "max_abs_std": gap_std,
            })
            if gap_mean > STRUCT_ATOL or gap_std > STRUCT_ATOL:
                reusable = False
    published = []
    for fold_id, held in enumerate(folds):
        path = RUNS / f"exp_b12/seed_42/arm_embeddings_only_fold{fold_id}.npz"
        if not path.exists():
            published.append(None)
            continue
        held_hard = held[np.asarray(hard, dtype=bool)[held]]
        logits = CrossCandidateAttentionNet(dict(np.load(path))).forward_numpy(embeddings[held_hard])
        choice = np.argmax(logits, axis=1)
        hits = labels[held_hard, choice] >= 0.5
        published.append((int(hits.sum()), int(hits.size)))
    published_match = published == list(PUBLISHED_ARM_A_HARD)
    if not published_match:
        reusable = False
    return {
        "reusable": reusable,
        "published_arm_a_hard": published,
        "published_arm_a_matches": published_match,
        "struct_gaps": gaps,
        "struct_atol": STRUCT_ATOL,
    }


def load_saved_attention(
    struct: np.ndarray,
    embeddings: np.ndarray,
    seed: int,
    fold_id: int,
    held: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    held_emb = embeddings[held]
    weights_a = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_only_fold{fold_id}.npz"))
    logits_a = CrossCandidateAttentionNet(weights_a).forward_numpy(held_emb)
    weights_b = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_plus_g01_g03_fold{fold_id}.npz"))
    struct_norm = (struct[held] - weights_b["struct_mean"]) / np.maximum(weights_b["struct_std"], 1e-6)
    features_b = np.concatenate([held_emb, struct_norm], axis=-1)
    logits_b = CrossCandidateAttentionNet(weights_b).forward_numpy(features_b)
    return np.asarray(logits_a, dtype=np.float32), np.asarray(logits_b, dtype=np.float32)


def score_rows(
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    weights: Sequence[float],
    temperature: float,
    rho: float,
    rows: np.ndarray,
) -> Dict[str, Any]:
    scored = score_aligned(
        views, labels, hard, bank, parses, eligible, table,
        weights, temperature, rho, rows,
    )
    scored.pop("choices", None)
    return scored


def select_configuration(
    configs: Sequence[Dict[str, Any]],
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    folds: Sequence[np.ndarray],
    excluded: Sequence[int],
) -> Dict[str, Any]:
    """Cross-fit every config on the inner folds that remain inside this fit partition."""
    banned = {int(fold_id) for fold_id in excluded}
    inners = [fold_id for fold_id in range(N_FOLDS) if fold_id not in banned]
    pooled: List[Dict[str, Any]] = []
    for config in configs:
        hard_correct = all_correct = hard_total = all_total = absent = 0
        nll_sum = 0.0
        for inner in inners:
            fit_rows = complement_rows(folds, list(banned | {inner}))
            eval_rows = folds[inner]
            normalized, _scales = normalize_views(
                views, config["normalization"], eligible, rows=fit_rows,
            )
            _weights, _info = fit_aligned_weights(
                normalized, hard, table, config["temperature"], config["rho"], fit_rows,
            )
            scored = score_rows(
                normalized, labels, hard, bank, parses, eligible, table,
                _weights, config["temperature"], config["rho"], eval_rows,
            )
            hard_correct += int(scored["hard_correct"])
            hard_total += int(scored["hard_total"])
            all_correct += int(scored["all_correct"])
            all_total += int(scored["all_total"])
            nll_sum += float(scored["nll_sum"])
            absent += int(scored["absent_gold_clusters"])
        pooled.append({
            "name": config["name"],
            "arm": config["arm"],
            "hard_correct": hard_correct,
            "hard_total": hard_total,
            "all_correct": all_correct,
            "all_total": all_total,
            "nll_sum": nll_sum,
            "nll_count": all_total,
            "absent_gold_clusters": absent,
        })
    index = select_config(pooled)
    return {"index": index, "config": configs[index], "pooled": pooled}


def fit_selected(
    config: Dict[str, Any],
    views: Sequence[np.ndarray],
    hard: np.ndarray,
    table: Dict[str, np.ndarray],
    eligible: np.ndarray,
    fit_rows: np.ndarray,
) -> Tuple[np.ndarray, Optional[List[float]], Dict[str, Any], List[np.ndarray]]:
    normalized, scales = normalize_views(views, config["normalization"], eligible, rows=fit_rows)
    weights, info = fit_aligned_weights(
        normalized, hard, table, config["temperature"], config["rho"], fit_rows,
    )
    return weights, scales, info, normalized


def decode_locked(
    config: Dict[str, Any],
    weights: Sequence[float],
    scales: Optional[Sequence[float]],
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if config["normalization"] == "fold_OOF_global_scale" and not scales:
        raise RuntimeError("global scale must come from the training partition")
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    normalized, _scales = normalize_views(views, config["normalization"], eligible, scales=scales)
    scored = score_aligned(
        normalized, labels, hard, bank, parses, eligible, table,
        weights, config["temperature"], config["rho"], np.arange(len(bank)),
    )
    choices = scored.pop("choices")
    return choices, scored


def retention_decision(confirmation: Dict[str, Any], contract: Dict[str, Any]) -> Dict[str, Any]:
    rules = contract["nested_protocol"]["final_validation_retention"]
    paired_hard = [float(item["hard_delta"]) for item in confirmation["paired"]]
    paired_all = [float(item["all_delta"]) for item in confirmation["paired"]]
    mean_hard = float(np.mean(paired_hard))
    mean_all = float(np.mean(paired_all))
    positive = int(sum(value > 0.0 for value in paired_hard))
    final_hard = int(confirmation["ensemble"]["hard_correct"])
    final_all = int(confirmation["ensemble"]["all_correct"])
    gates = {
        "final_hard_ge_302": {"value": final_hard, "passed": bool(final_hard >= int(rules["minimum_final_hard_correct"]))},
        "final_all_ge_743": {"value": final_all, "passed": bool(final_all >= int(rules["minimum_final_all_correct"]))},
        "paired_hard_mean_ge_1": {"value": mean_hard, "passed": bool(mean_hard >= float(rules["paired_seed_minimum_mean_hard_delta_questions"]))},
        "paired_hard_positive_ge_2": {"value": positive, "passed": bool(positive >= int(rules["paired_seed_minimum_positive_count"]))},
        "paired_all_mean_ge_0": {"value": mean_all, "passed": bool(mean_all >= float(rules["paired_seed_minimum_mean_all_delta_questions"]))},
    }
    retained = all(item["passed"] for item in gates.values())
    return {
        "gates": gates,
        "retained": retained,
        "verdict": "RETAINED" if retained else "VALIDATION_NOT_RETAINED",
        "historical_hard_delta": final_hard - HISTORICAL_HARD,
        "historical_all_delta": final_all - HISTORICAL_ALL,
        "strict_hard_delta": final_hard - B20_STRICT_HARD,
        "strict_all_delta": final_all - B20_STRICT_ALL,
    }


def run_validation_confirmation(
    lock: Dict[str, Any],
    train_pack: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """One frozen candidate. The caller writes the train lock before this function."""
    val = SplitData("validation")
    val_hard = np.asarray(val.hard, dtype=bool)
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    tree_val = np.asarray(np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")["seed_42"])
    ensemble_views = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], tree_val]
    historical = score_choice(system_choice(HISTORICAL_WEIGHTS, ensemble_views, val.bank), val.labels, val_hard)
    strict = score_choice(system_choice(B20_STRICT_WEIGHTS, ensemble_views, val.bank), val.labels, val_hard)
    if historical["val_hard"]["correct"] != HISTORICAL_HARD or historical["val_all"]["correct"] != HISTORICAL_ALL:
        raise RuntimeError(
            f"historical replay is {historical['val_hard']['correct']}/{historical['val_all']['correct']}, "
            f"expected {HISTORICAL_HARD}/{HISTORICAL_ALL}"
        )
    if strict["val_hard"]["correct"] != B20_STRICT_HARD or strict["val_all"]["correct"] != B20_STRICT_ALL:
        raise RuntimeError(
            f"strict replay is {strict['val_hard']['correct']}/{strict['val_all']['correct']}, "
            f"expected {B20_STRICT_HARD}/{B20_STRICT_ALL}"
        )

    config = lock["config"]
    labels = train_pack["labels"]
    hard = train_pack["hard"]
    table = train_pack["table"]
    eligible = train_pack["eligible"]
    started = time.perf_counter()
    ensemble_choice, ensemble_score = decode_locked(
        config, lock["weights"], lock["scales"], ensemble_views, val.labels, val_hard, val.bank,
    )
    decode_seconds = time.perf_counter() - started
    reference_weights = fit_three_weights(
        train_pack["mean_a"], train_pack["mean_b"], train_pack["tree"], labels, hard,
    )
    reference_choice = np.asarray(system_choice(reference_weights, ensemble_views, val.bank), dtype=np.int64)
    reference_score = count_choice(reference_choice, val.labels, val_hard)
    historical_choice = np.asarray(system_choice(HISTORICAL_WEIGHTS, ensemble_views, val.bank), dtype=np.int64)

    paired = []
    for seed in SEEDS:
        seed_views = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], tree_val]
        old_weights = fit_three_weights(
            train_pack["per_seed_a"][seed], train_pack["per_seed_b"][seed], train_pack["tree"], labels, hard,
        )
        old_score = count_choice(system_choice(old_weights, seed_views, val.bank), val.labels, val_hard)
        new_weights, new_scales, new_info, _normalized = fit_selected(
            config,
            [train_pack["per_seed_a"][seed], train_pack["per_seed_b"][seed], train_pack["tree"]],
            hard, table, eligible, np.arange(labels.shape[0]),
        )
        new_choice, new_score = decode_locked(
            config, new_weights, new_scales, seed_views, val.labels, val_hard, val.bank,
        )
        paired.append({
            "seed": int(seed),
            "old_hard": old_score["hard_correct"],
            "new_hard": new_score["hard_correct"],
            "hard_delta": int(new_score["hard_correct"] - old_score["hard_correct"]),
            "old_all": old_score["all_correct"],
            "new_all": new_score["all_correct"],
            "all_delta": int(new_score["all_correct"] - old_score["all_correct"]),
            "weights": [float(value) for value in new_weights],
            "scales": new_scales,
            "fit_loss": new_info["loss"],
            "choices_head": new_choice[:8].tolist(),
        })

    prediction_path = EXP_DIR / "exp_b25_validation_predictions.json"
    score_path = EXP_DIR / "exp_b25_validation_score.json"
    write_official_predictions(prediction_path, val.bank, ensemble_choice)
    official = official_score(prediction_path, score_path)
    if int(official["correct"]) != int(ensemble_score["all_correct"]):
        raise RuntimeError(
            f"official all-count {official['correct']} does not match in-process {ensemble_score['all_correct']}"
        )
    same_protocol_hits = val.labels[np.arange(len(val.bank)), reference_choice] >= 0.5
    new_hits = val.labels[np.arange(len(val.bank)), ensemble_choice] >= 0.5
    historical_hits = val.labels[np.arange(len(val.bank)), historical_choice] >= 0.5
    asset_bytes = int(contract["baseline"]["historical"]["recorded_uncompressed_bytes"])
    lock_bytes = (EXP_DIR / "downstream_decoder_lock.json").stat().st_size
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
        "same_protocol_reference": reference_score,
        "paired": paired,
        "flips_vs_same_protocol": {
            "corrected": int((new_hits & ~same_protocol_hits).sum()),
            "spoiled": int((~new_hits & same_protocol_hits).sum()),
        },
        "flips_vs_historical": {
            "corrected": int((new_hits & ~historical_hits).sum()),
            "spoiled": int((~new_hits & historical_hits).sum()),
        },
        "official_score": official,
        "in_process_decode_seconds": decode_seconds,
        "asset_preaudit": {
            "scope": "local in-process decode plus the unchanged B20 assets and the decoder lock",
            "not_a_fresh_predict_process": True,
            "not_a_xeon_certificate": True,
            "b20_uncompressed_bytes": asset_bytes,
            "lock_bytes": int(lock_bytes),
            "combined_bytes": int(asset_bytes + lock_bytes),
        },
        "validation_configurations": 4,
    }
    confirmation["retention_gate"] = retention_decision(confirmation, contract)
    return confirmation


def assert_seed_streams() -> int:
    fold_seeds = set()
    for seed in SEEDS:
        for fold_id in range(N_FOLDS):
            derived = fold_model_seeds(seed, fold_id)
            fold_seeds.update(derived.values())
    nested = []
    for variant in (0, 1):
        for seed in SEEDS:
            for outer in range(N_FOLDS):
                for inner in range(N_FOLDS):
                    derived = nested_attention_seeds(variant, seed, outer, inner)
                    nested.extend(derived.values())
    if len(nested) != len(set(nested)) or fold_seeds.intersection(nested):
        raise RuntimeError("nested attention seeds collide with each other or with fold seeds")
    return len(nested)


def train_nested_oof(
    embeddings: np.ndarray,
    struct: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    folds: Sequence[np.ndarray],
    device,
) -> Dict[str, Any]:
    count = len(labels)
    nested_a = np.zeros((N_FOLDS, count, 4), dtype=np.float32)
    nested_b = np.zeros((N_FOLDS, count, 4), dtype=np.float32)
    nested_tree = np.zeros((N_FOLDS, count, 4), dtype=np.float32)
    filled = np.zeros((N_FOLDS, count), dtype=bool)
    attention_steps = 0
    attention_seconds = 0.0
    tree_seconds = 0.0
    cells = []
    cell_id = 0
    for outer in range(N_FOLDS):
        for inner in range(N_FOLDS):
            if inner == outer:
                continue
            cell_id += 1
            fit_rows = complement_rows(folds, [outer, inner])
            held = folds[inner]
            if set(int(row) for row in fit_rows).intersection(int(row) for row in held):
                raise RuntimeError(f"outer {outer} inner {inner} leaked into its fit rows")
            tree, seconds = train_tree(embeddings, struct, labels, fit_rows, held)
            nested_tree[outer, held] = tree
            tree_seconds += seconds
            sum_a = np.zeros((held.size, 4), dtype=np.float64)
            sum_b = np.zeros((held.size, 4), dtype=np.float64)
            for seed in SEEDS:
                for variant, arm, bucket in ((0, "arm_a", sum_a), (1, "arm_b", sum_b)):
                    logits, steps, fit_seconds = train_attention(
                        embeddings, struct, labels, hard, fit_rows, held,
                        nested_attention_seeds(variant, seed, outer, inner), device, arm,
                    )
                    bucket += logits
                    attention_steps += steps
                    attention_seconds += fit_seconds
            nested_a[outer, held] = (sum_a / len(SEEDS)).astype(np.float32)
            nested_b[outer, held] = (sum_b / len(SEEDS)).astype(np.float32)
            filled[outer, held] = True
            cells.append({
                "outer": outer,
                "inner": inner,
                "fit_rows_sha256": rows_sha256(fit_rows),
                "held_rows_sha256": rows_sha256(held),
                "fit_questions": int(fit_rows.size),
                "held_questions": int(held.size),
            })
            print(
                f"  nested {cell_id}/20 outer {outer} inner {inner}: "
                f"tree {seconds:.1f}s, attention cell {attention_seconds:.1f}s cumulative",
                flush=True,
            )
    return {
        "a": nested_a,
        "b": nested_b,
        "tree": nested_tree,
        "filled": filled,
        "attention_steps": attention_steps,
        "attention_seconds": attention_seconds,
        "tree_seconds": tree_seconds,
        "cells": cells,
    }


def train_outer_oof(
    embeddings: np.ndarray,
    struct: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    folds: Sequence[np.ndarray],
    device,
    reuse_b12: bool,
) -> Dict[str, Any]:
    count = len(labels)
    per_a = {int(seed): np.zeros((count, 4), dtype=np.float32) for seed in SEEDS}
    per_b = {int(seed): np.zeros((count, 4), dtype=np.float32) for seed in SEEDS}
    tree = np.zeros((count, 4), dtype=np.float32)
    attention_steps = 0
    attention_seconds = 0.0
    tree_seconds = 0.0
    for outer, held in enumerate(folds):
        fit_rows = complement_rows(folds, [outer])
        if reuse_b12:
            for seed in SEEDS:
                logits_a, logits_b = load_saved_attention(struct, embeddings, int(seed), outer, held)
                per_a[int(seed)][held] = logits_a
                per_b[int(seed)][held] = logits_b
        else:
            for seed in SEEDS:
                derived = fold_model_seeds(int(seed), outer)
                logits_a, steps_a, seconds_a = train_attention(
                    embeddings, struct, labels, hard, fit_rows, held, derived, device, "arm_a",
                )
                logits_b, steps_b, seconds_b = train_attention(
                    embeddings, struct, labels, hard, fit_rows, held, derived, device, "arm_b",
                )
                per_a[int(seed)][held] = logits_a
                per_b[int(seed)][held] = logits_b
                attention_steps += steps_a + steps_b
                attention_seconds += seconds_a + seconds_b
        held_tree, seconds = train_tree(embeddings, struct, labels, fit_rows, held)
        tree[held] = held_tree
        tree_seconds += seconds
        print(f"  outer complement fold {outer}: tree {seconds:.1f}s", flush=True)
    mean_a = np.mean([per_a[int(seed)] for seed in SEEDS], axis=0).astype(np.float32)
    mean_b = np.mean([per_b[int(seed)] for seed in SEEDS], axis=0).astype(np.float32)
    return {
        "per_seed_a": per_a,
        "per_seed_b": per_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "tree": tree,
        "attention_steps": attention_steps,
        "attention_seconds": attention_seconds,
        "tree_seconds": tree_seconds,
    }


def score_held_config(
    config: Dict[str, Any],
    weights: Sequence[float],
    scales: Optional[Sequence[float]],
    held_views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    if config["normalization"] == "fold_OOF_global_scale" and not scales:
        raise RuntimeError("held-out global scale is missing its fit-partition RMS")
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    normalized, _scales = normalize_views(
        held_views, config["normalization"], eligible, scales=scales,
    )
    scored = score_aligned(
        normalized, labels, hard, bank, parses, eligible, table,
        weights, config["temperature"], config["rho"], np.arange(len(bank)),
    )
    return {
        "hard_correct": int(scored["hard_correct"]),
        "hard_total": int(scored["hard_total"]),
        "all_correct": int(scored["all_correct"]),
        "all_total": int(scored["all_total"]),
        "nll_sum": float(scored["nll_sum"]),
        "absent_gold_clusters": int(scored["absent_gold_clusters"]),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def run_exp_b25() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B25: align fusion fit with the deployed decoder")
    print("=" * 78, flush=True)
    started = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "experiment": "EXP-B25",
        "protocol": PROTOCOL,
        "validation_queries": 0,
        "validation_opened": False,
        "test_references_opened": False,
        "written_to_model_bundle": False,
        "candidate_submission_updated": False,
        "validation_confirmation": None,
        "verdict": "FAILED",
    }
    try:
        contract = load_contract()
        verify_sources(contract)
        seed_values = assert_seed_streams()
        configs = b25_configs(contract)
        device = configure_determinism()
        environment = environment_record(device)
        environment["executable"] = sys.executable
        print(f"Interpreter: {sys.executable}", flush=True)
        train = SplitData("train")
        if len(train.bank) != 4755 or int(np.asarray(train.hard).sum()) != 2114:
            raise RuntimeError("train split is not the registered 4755/2114 partition")
        labels = train.labels
        hard = np.asarray(train.hard, dtype=bool)
        folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
        eligible, parses = question_eligibility(train.bank)
        table = build_cluster_table(parses, eligible, labels)
        struct, _g01, _g03 = reasoning_struct(train)
        if struct.shape[-1] != 16:
            raise RuntimeError(f"Arm B structured block has width {struct.shape[-1]}")

        print("Checking saved B12 fold models", flush=True)
        b12_check = verify_saved_folds(struct, train.emb, labels, hard, folds)
        print(
            f"  B12 reusable={b12_check['reusable']} "
            f"arm A published match={b12_check['published_arm_a_matches']}",
            flush=True,
        )

        print("Training nested OOF (120 attention models, 20 trees)", flush=True)
        nested = train_nested_oof(train.emb, struct, labels, hard, folds, device)
        print("Scoring complement-only outer models", flush=True)
        outer = train_outer_oof(
            train.emb, struct, labels, hard, folds, device, bool(b12_check["reusable"]),
        )

        reference_folds = []
        selected_folds = []
        fold_records = []
        for fold_id, held in enumerate(folds):
            complement = complement_rows(folds, [fold_id])
            if not bool(nested["filled"][fold_id, complement].all()):
                raise RuntimeError(f"outer {fold_id} is missing nested OOF rows")
            nested_views = [nested["a"][fold_id], nested["b"][fold_id], nested["tree"][fold_id]]
            print(f"Selecting decoder on outer {fold_id}", flush=True)
            selection = select_configuration(
                configs, nested_views, labels, hard, train.bank, parses, eligible, table,
                folds, [fold_id],
            )
            config = selection["config"]
            weights, scales, info, _normalized = fit_selected(
                config, nested_views, hard, table, eligible, complement,
            )
            held_views = [outer["mean_a"][held], outer["mean_b"][held], outer["tree"][held]]
            held_bank = slice_bank(train.bank, held)
            selected = score_held_config(
                config, weights, scales, held_views, labels[held], hard[held], held_bank,
            )
            reference_weights = fit_three_weights(
                nested_views[0][complement], nested_views[1][complement], nested_views[2][complement],
                labels[complement], hard[complement],
            )
            reference_choice = system_choice(reference_weights, held_views, held_bank)
            reference = count_choice(reference_choice, labels[held], hard[held])
            reference_folds.append(reference)
            selected_folds.append(selected)
            fold_records.append({
                "outer_fold": fold_id,
                "fit_rows_sha256": rows_sha256(complement),
                "held_rows_sha256": rows_sha256(held),
                "selected_config": config,
                "selection": selection["pooled"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "fit_loss": info["loss"],
                "fit_iterations": info["iterations"],
                "absent_gold_clusters_fit": info["absent_gold_clusters"],
                "reference_weights": [float(value) for value in reference_weights],
                "reference": reference,
                "selected": selected,
            })
            print(
                f"  outer {fold_id}: {config['name']} "
                f"hard {selected['hard_correct']}/{selected['hard_total']} "
                f"vs reference {reference['hard_correct']}/{reference['hard_total']}",
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

        oof_path = EXP_DIR / "oof_logits.npz"
        np.savez(
            oof_path,
            nested_mean_a=nested["a"],
            nested_mean_b=nested["b"],
            nested_tree=nested["tree"],
            full_mean_a=outer["mean_a"],
            full_mean_b=outer["mean_b"],
            full_tree=outer["tree"],
            **{f"full_a_seed_{seed}": outer["per_seed_a"][int(seed)] for seed in SEEDS},
            **{f"full_b_seed_{seed}": outer["per_seed_b"][int(seed)] for seed in SEEDS},
        )
        oof_sha = sha256_file(oof_path)
        manifest = {
            "experiment": "EXP-B25",
            "source_sha256": contract["source_sha256"],
            "b12_reuse": b12_check,
            "nested_cells": nested["cells"],
            "seed_formula": "warmup=1000000+100000*variant+10000*s+10*outer+inner; finetune=warmup+1000",
            "variant_codes": {"arm_a": 0, "arm_b": 1},
            "seeds": list(SEEDS),
            "three_seed_average_before_view_normalization": True,
            "base_recipe": "B12 warmup on all fit rows, finetune on hard fit rows, no sample weights",
            "hparams": HPARAMS,
            "preprocessing_fit_scope": "statistics and SVD fit on the complement of the excluded folds only",
            "oof_sha256": oof_sha,
        }
        manifest_sha = write_json(EXP_DIR / "oof_manifest.json", manifest)
        register_seeds("EXP-B25", [{
            "arm": "nested_decoder_oof",
            "seed": "42/43/44",
            "member": "nested_and_outer_oof",
            "derived": {
                "nested_formula": manifest["seed_formula"],
                "attention_models": 120,
                "nested_trees": 20,
                "outer_trees": 5,
                "unique_nested_seed_values": seed_values,
                "b12_fold_models_reused": bool(b12_check["reusable"]),
            },
            "file": rel(oof_path),
            "sha256": oof_sha,
        }])

        audit.update({
            "environment": environment,
            "source_hashes_matched_at_start": True,
            "b12_reuse": {
                "reusable": b12_check["reusable"],
                "published_arm_a_matches": b12_check["published_arm_a_matches"],
                "published_arm_a_hard": b12_check["published_arm_a_hard"],
                "struct_atol": STRUCT_ATOL,
            },
            "training": {
                "nested_attention_models": 120,
                "nested_trees": 20,
                "outer_trees": 5,
                "nested_attention_steps": nested["attention_steps"],
                "nested_attention_seconds": nested["attention_seconds"],
                "nested_tree_seconds": nested["tree_seconds"],
                "outer_attention_steps": outer["attention_steps"],
                "outer_attention_seconds": outer["attention_seconds"],
                "outer_tree_seconds": outer["tree_seconds"],
                "b12_fold_models_reused": bool(b12_check["reusable"]),
                "new_base_training_recipe": False,
            },
            "configurations": [config["name"] for config in configs],
            "outer_folds": fold_records,
            "internal_gate": gate,
            "oof_sha256": oof_sha,
            "manifest_sha256": manifest_sha,
        })

        if not gate["passed"]:
            audit["verdict"] = "STOPPED_INTERNAL_GATE"
            audit["validation_confirmation"] = None
            print("Internal gate did not pass. Validation stays closed.", flush=True)
        else:
            print("Selecting the full-train decoder lock", flush=True)
            full_views = [outer["mean_a"], outer["mean_b"], outer["tree"]]
            final = select_configuration(
                configs, full_views, labels, hard, train.bank, parses, eligible, table, folds, [],
            )
            all_rows = np.arange(len(train.bank))
            weights, scales, info, _normalized = fit_selected(
                final["config"], full_views, hard, table, eligible, all_rows,
            )
            lock = {
                "experiment": "EXP-B25",
                "purpose": "full-train decoder for final refit and validation; not an outer-fold input",
                "config": final["config"],
                "weights": [float(value) for value in weights],
                "scales": scales,
                "weight_fit": info,
                "selection": final["pooled"],
                "selected_index": final["index"],
                "fit_scope": "full-train inner OOF; weights and scale refit on all training rows",
                "source_sha256": contract["source_sha256"],
                "oof_sha256": oof_sha,
                "candidate_prediction_rule": "locked cluster decoder on the three-seed mean views",
            }
            lock_sha = write_json(EXP_DIR / "downstream_decoder_lock.json", lock)
            fold_lock_sha = write_json(EXP_DIR / "fold_local_locks.json", {"folds": fold_records})
            snapshot = {
                "chosen_config": final["config"],
                "fit_scope_manifest": manifest_sha,
                "source_sha256": contract["source_sha256"],
                "asset_sha256": {"decoder_lock": lock_sha, "fold_local_locks": fold_lock_sha, "oof": oof_sha},
                "internal_gate": gate,
                "candidate_prediction_rule": lock["candidate_prediction_rule"],
            }
            snapshot_sha = write_json(EXP_DIR / "validation_snapshot.json", snapshot)
            audit["decoder_lock"] = {"sha256": lock_sha, "config": final["config"], "weights": lock["weights"]}
            audit["fold_local_lock_sha256"] = fold_lock_sha
            audit["validation_snapshot_sha256"] = snapshot_sha
            audit["validation_opened"] = True
            gate["validation_opened"] = True
            train_pack = {
                "labels": labels,
                "hard": hard,
                "table": table,
                "eligible": eligible,
                "mean_a": outer["mean_a"],
                "mean_b": outer["mean_b"],
                "tree": outer["tree"],
                "per_seed_a": outer["per_seed_a"],
                "per_seed_b": outer["per_seed_b"],
            }
            confirmation = run_validation_confirmation(lock, train_pack, contract)
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
    run_exp_b25()
