"""EXP-B23: reweight arm A's training loss from a nested OOF baseline.

Only arm A changes. The model, the two-phase recipe, and the unweighted
reference stay the same. Inside each outer training fold, question weights
come from an inner OOF that has not seen the question and has not seen the
outer held-out fold. Weights are 2 for an incorrect argmax, 0.5 for a correct
argmax on a question that is not train-hard, and 1 otherwise. They multiply
the training loss. They are not a resampling budget.

The internal gate compares seed-42 arm A argmax accuracy on held-out hard
questions. The unweighted reference is the saved EXP-B12 fold model. The
weighted model uses that fold model's seed, so initialization and batch order
match. A gain of at least 1 percentage point on the unweighted mean of the
five fold rates is required before any three-seed retrain or validation.
"""
from __future__ import annotations

import json
import os
import sys
import time
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

from experiments.exp_b07_cross_attn import HPARAMS, kfold_indices  # noqa: E402
from experiments.multiseed import (  # noqa: E402
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    configure_determinism,
    environment_record,
    fold_model_seeds,
    full_model_seeds,
    hits_of,
    register_seeds,
    rel,
    save_npz,
    train_model,
)
from models.cross_attn_net import CrossCandidateAttentionNet  # noqa: E402

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
GATE_SEED = 42
N_FOLDS = 5
MINING_SEED_BASE = 200_000
WEIGHT_INCORRECT = 2.0
WEIGHT_CORRECT_NOT_HARD = 0.5
WEIGHT_OTHER = 1.0
# Seed-42 embeddings-only fold models in exp_b12_seed_ensemble_audit.json.
PUBLISHED_BASELINE_HARD = ((345, 444), (330, 424), (289, 392), (325, 416), (351, 438))

EXP_DIR = RUNS / "exp_b23"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b23_hard_weighting_audit.json"
CONTRACT_PATH = AUDIT_PATH.parent / "private_configuration_omitted.json"


def question_loss_weights(correct: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """Per-question loss weights from an OOF argmax and the dataset hard flag."""
    correct = np.asarray(correct, dtype=bool)
    hard = np.asarray(hard, dtype=bool)
    if correct.shape != hard.shape or correct.ndim != 1:
        raise ValueError(f"correct/hard shape mismatch: {correct.shape} vs {hard.shape}")
    weights = np.full(correct.shape, WEIGHT_OTHER, dtype=np.float32)
    weights[~correct] = np.float32(WEIGHT_INCORRECT)
    weights[correct & ~hard] = np.float32(WEIGHT_CORRECT_NOT_HARD)
    return weights


def complement_rows(folds: Sequence[np.ndarray], excluded: Sequence[int]) -> np.ndarray:
    """Questions in every fold except the excluded fold ids, in fold order."""
    banned = {int(fold_id) for fold_id in excluded}
    parts = [folds[fold_id] for fold_id in range(len(folds)) if fold_id not in banned]
    if not parts:
        raise ValueError("complement rows are empty")
    return np.concatenate(parts)


def mining_seeds(seed: int, outer: int, inner: int) -> Dict[str, int]:
    warmup = MINING_SEED_BASE + 10000 * (int(seed) - GATE_SEED) + 100 * int(outer) + int(inner)
    return {"warmup": int(warmup), "finetune": int(warmup + 1000)}


def internal_gain_percentage_points(
    old_folds: Sequence[Tuple[int, int]],
    new_folds: Sequence[Tuple[int, int]],
) -> float:
    """Unweighted mean of fold hard-accuracy percentages, new minus old."""
    def mean_percent(folds: Sequence[Tuple[int, int]]) -> float:
        return float(np.mean([100.0 * correct / total for correct, total in folds]))

    return mean_percent(new_folds) - mean_percent(old_folds)


def internal_gate_passed(gain: float, minimum: float) -> bool:
    return bool(np.isfinite(gain) and float(gain) >= float(minimum))


def expected_optimizer_steps(n_rows: int, epochs: int, batch_size: int) -> int:
    if n_rows <= 0:
        raise ValueError("optimizer step count requires at least one row")
    return int(epochs) * ((int(n_rows) + int(batch_size) - 1) // int(batch_size))


def mean_percent(counts: Sequence[Tuple[int, int]]) -> float:
    return float(np.mean([100.0 * correct / total for correct, total in counts]))


def pooled_count(counts: Sequence[Tuple[int, int]]) -> Dict[str, int]:
    correct = int(sum(item[0] for item in counts))
    total = int(sum(item[1] for item in counts))
    return {"correct": correct, "total": total}


def weight_histogram(weights: np.ndarray) -> Dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    incorrect = int(np.isclose(weights, WEIGHT_INCORRECT).sum())
    correct_easy = int(np.isclose(weights, WEIGHT_CORRECT_NOT_HARD).sum())
    other = int(np.isclose(weights, WEIGHT_OTHER).sum())
    if incorrect + correct_easy + other != int(weights.size):
        raise RuntimeError("question weights fell outside the three registered buckets")
    return {
        "questions": int(weights.size),
        "incorrect": incorrect,
        "correct_not_hard": correct_easy,
        "other_correct_hard": other,
        "weight_sum": float(weights.sum()),
        "mean_weight": float(weights.mean()),
    }


def score_saved_attention(path: Path, features: np.ndarray) -> np.ndarray:
    weights = dict(np.load(path))
    return CrossCandidateAttentionNet(weights).forward_numpy(features)


def argmax_count(logits: np.ndarray, labels: np.ndarray) -> Tuple[int, int]:
    choice = np.argmax(logits, axis=1)
    hits = hits_of(choice, labels)
    return int(hits.sum()), int(hits.size)


def load_b23_rules() -> Dict[str, float]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    spec = payload["followup_batches"]["EXP-B23"]
    weights = spec["sampling_loss_weights"]
    if float(weights["training_prediction_incorrect"]) != WEIGHT_INCORRECT:
        raise RuntimeError("contract incorrect-question weight is not 2")
    if float(weights["training_prediction_correct_and_not_hard"]) != WEIGHT_CORRECT_NOT_HARD:
        raise RuntimeError("contract correct-and-not-hard weight is not 0.5")
    if float(weights["other"]) != WEIGHT_OTHER:
        raise RuntimeError("contract other weight is not 1")
    minimum = float(spec["train_internal_min_heldout_hard_gain_percentage_points"])
    if minimum != 1.0:
        raise RuntimeError("contract internal gate is not 1 percentage point")
    return {"minimum_percentage_points": minimum}


def preflight_baseline(train: SplitData, folds: Sequence[np.ndarray]) -> List[Tuple[int, int]]:
    """Rescore the saved seed-42 arm A fold models and require the published counts."""
    hard = np.asarray(train.hard, dtype=bool)
    counts: List[Tuple[int, int]] = []
    for fold_id, held in enumerate(folds):
        held_hard = held[hard[held]]
        path = RUNS / f"exp_b12/seed_{GATE_SEED}/arm_embeddings_only_fold{fold_id}.npz"
        logits = score_saved_attention(path, train.emb[held_hard])
        correct, total = argmax_count(logits, train.labels[held_hard])
        expected = PUBLISHED_BASELINE_HARD[fold_id]
        if (correct, total) != expected:
            raise RuntimeError(
                f"fold {fold_id} unweighted arm A is {correct}/{total}, "
                f"published baseline is {expected[0]}/{expected[1]}"
            )
        counts.append((correct, total))
        print(f"  baseline fold {fold_id}: {correct}/{total}")
    return counts


def train_uniform_arm(
    features: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    seeds: Dict[str, int],
    device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    local_hard = np.flatnonzero(np.asarray(hard, dtype=bool))
    if local_hard.size == 0:
        raise RuntimeError("uniform arm A fit has no hard questions")
    return train_model(
        features, labels, features[local_hard], labels[local_hard], seeds, device,
    )


def train_weighted_arm(
    features: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    question_weight: np.ndarray,
    seeds: Dict[str, int],
    device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    local_hard = np.flatnonzero(np.asarray(hard, dtype=bool))
    if local_hard.size == 0:
        raise RuntimeError("weighted arm A fit has no hard questions")
    if len(question_weight) != len(features):
        raise RuntimeError("question weights are not aligned with the fit rows")
    state, stats = train_model(
        features,
        labels,
        features[local_hard],
        labels[local_hard],
        seeds,
        device,
        sample_weight_warm=np.asarray(question_weight, dtype=np.float32),
        sample_weight_fine=np.asarray(question_weight, dtype=np.float32)[local_hard],
    )
    warm_steps = int(stats["warmup"]["optimizer_steps"])
    fine_steps = int(stats["finetune"]["optimizer_steps"])
    expected_warm = expected_optimizer_steps(len(features), HPARAMS["warmup_epochs"], HPARAMS["batch_size"])
    expected_fine = expected_optimizer_steps(int(local_hard.size), HPARAMS["finetune_epochs"], HPARAMS["batch_size"])
    if warm_steps != expected_warm or fine_steps != expected_fine:
        raise RuntimeError(
            f"weighted optimizer steps {warm_steps}+{fine_steps} "
            f"do not match the unweighted budget {expected_warm}+{expected_fine}"
        )
    if abs(float(stats["warmup"]["weight_sum"]) - float(np.sum(question_weight))) > 1e-3:
        raise RuntimeError("recorded warmup weight sum does not match the question weights")
    return state, stats


def fit_nested_weighted_oof(
    train: SplitData,
    seed: int,
    device,
    registry: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Train one weighted arm A per outer fold. Mining never sees that fold."""
    folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
    hard = np.asarray(train.hard, dtype=bool)
    labels = train.labels
    oof = np.zeros((len(train.bank), 4), dtype=np.float32)
    fold_rows = []
    weighted_counts: List[Tuple[int, int]] = []
    mining_seconds = 0.0
    weighted_seconds = 0.0
    mining_steps = 0
    comparison_steps = 0
    for outer, held in enumerate(folds):
        outer_rows = complement_rows(folds, [outer])
        held_set = set(int(index) for index in held.tolist())
        if held_set.intersection(int(index) for index in outer_rows.tolist()):
            raise RuntimeError(f"outer fold {outer} leaked into its training rows")
        correct = np.zeros(len(train.bank), dtype=bool)
        covered = np.zeros(len(train.bank), dtype=bool)
        outer_mining_steps = 0
        outer_mining_seconds = 0.0
        for inner in range(N_FOLDS):
            if inner == outer:
                continue
            mine_rows = complement_rows(folds, [outer, inner])
            inner_rows = folds[inner]
            if held_set.intersection(int(index) for index in mine_rows.tolist()):
                raise RuntimeError(f"outer fold {outer} leaked into mining fold {inner}")
            if set(int(index) for index in inner_rows.tolist()).intersection(int(index) for index in mine_rows.tolist()):
                raise RuntimeError(f"inner fold {inner} was scored by a model that trained on it")
            seeds = mining_seeds(seed, outer, inner)
            started = time.perf_counter()
            state, stats = train_uniform_arm(
                train.emb[mine_rows], labels[mine_rows], hard[mine_rows], seeds, device,
            )
            elapsed = time.perf_counter() - started
            logits = CrossCandidateAttentionNet(state).forward_numpy(train.emb[inner_rows])
            correct[inner_rows] = hits_of(np.argmax(logits, axis=1), labels[inner_rows])
            covered[inner_rows] = True
            steps = int(stats["warmup"]["optimizer_steps"]) + int(stats["finetune"]["optimizer_steps"])
            outer_mining_steps += steps
            outer_mining_seconds += elapsed
            mining_steps += steps
            mining_seconds += elapsed
            print(f"  [seed {seed} outer {outer}] mine inner {inner}: {elapsed:.1f}s, {steps} steps")
        if int(covered[outer_rows].sum()) != int(outer_rows.size) or bool(covered[held].any()):
            raise RuntimeError(f"outer fold {outer} OOF coverage is not exactly the training rows")
        question_weight = question_loss_weights(correct[outer_rows], hard[outer_rows])
        comparison = fold_model_seeds(seed, outer)
        started = time.perf_counter()
        state, stats = train_weighted_arm(
            train.emb[outer_rows], labels[outer_rows], hard[outer_rows], question_weight, comparison, device,
        )
        elapsed = time.perf_counter() - started
        weighted_seconds += elapsed
        held_logits = CrossCandidateAttentionNet(state).forward_numpy(train.emb[held])
        oof[held] = held_logits
        held_hard = held[hard[held]]
        hard_positions = np.flatnonzero(hard[held])
        correct_n, total_n = argmax_count(held_logits[hard_positions], labels[held_hard])
        weighted_counts.append((correct_n, total_n))
        warm_steps = int(stats["warmup"]["optimizer_steps"])
        fine_steps = int(stats["finetune"]["optimizer_steps"])
        comparison_steps += warm_steps + fine_steps
        path = EXP_DIR / f"seed_{seed}" / f"arm_a_weighted_fold{outer}.npz"
        digest = save_npz(path, state)
        registry.append({
            "arm": "embeddings_only_weighted",
            "seed": int(seed),
            "member": f"fold{outer}",
            "derived": comparison,
            "file": rel(path),
            "sha256": digest,
            "role": "outer-fold weighted arm A; loss weights mined without this fold",
        })
        histogram = weight_histogram(question_weight)
        fold_rows.append({
            "outer_fold": int(outer),
            "weighted_hard": {"correct": correct_n, "total": total_n},
            "question_weights": histogram,
            "finetune_weight_sum": float(question_weight[hard[outer_rows]].sum()),
            "finetune_rows": int(hard[outer_rows].sum()),
            "warmup_optimizer_steps": warm_steps,
            "finetune_optimizer_steps": fine_steps,
            "mining_optimizer_steps": int(outer_mining_steps),
            "mining_fit_seconds": outer_mining_seconds,
            "weighted_fit_seconds": elapsed,
            "checkpoint": rel(path),
            "sha256": digest,
        })
        print(
            f"  [seed {seed} outer {outer}] weighted arm A {correct_n}/{total_n} "
            f"weights 2/0.5/1 = {histogram['incorrect']}/{histogram['correct_not_hard']}/"
            f"{histogram['other_correct_hard']} ({elapsed:.1f}s)"
        )
    oof_path = EXP_DIR / f"seed_{seed}" / "arm_a_weighted_oof_logits.npy"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)
    report = {
        "seed": int(seed),
        "folds": fold_rows,
        "weighted_hard_counts": weighted_counts,
        "mining_fit_seconds": mining_seconds,
        "weighted_fit_seconds": weighted_seconds,
        "mining_optimizer_steps": int(mining_steps),
        "comparison_optimizer_steps": int(comparison_steps),
        "oof_logits": rel(oof_path),
    }
    return report, oof


def run_internal_gate(
    train: SplitData,
    device,
    minimum: float,
    registry: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray]:
    print("\n--- Internal five-fold gate (seed 42 arm A, validation closed) ---")
    folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
    started = time.perf_counter()
    print("Rescoring the saved unweighted arm A fold models")
    baseline_counts = preflight_baseline(train, folds)
    report, oof = fit_nested_weighted_oof(train, GATE_SEED, device, registry)
    weighted_counts = report.pop("weighted_hard_counts")
    gain = internal_gain_percentage_points(baseline_counts, weighted_counts)
    passed = internal_gate_passed(gain, minimum)
    print(f"Internal held-out hard mean gain: {gain:+.4f} percentage points (minimum {minimum:.1f})")
    gate = {
        "passed": passed,
        "gain_percentage_points": gain,
        "minimum_percentage_points": minimum,
        "selection": "arm A argmax on held-out hard questions",
        "hard_definition": "train.hard disagreement flag",
        "mining_prediction": "inner OOF argmax hits a correct candidate",
        "loss": "per-question weighted mean of listwise plus lambda_bce times summed BCE",
        "uniform_weights_match_the_unweighted_mean": True,
        "comparison_seed_formula": "fold_model_seeds(seed, outer), same initialization as the saved B12 fold model",
        "mining_seed_formula": "warmup = 200000 + 10000*(seed-42) + 100*outer + inner; finetune = warmup + 1000",
        "baseline_source": "dev/runs/exp_b12/seed_42/arm_embeddings_only_fold{k}.npz",
        "old_hard_mean_percent": mean_percent(baseline_counts),
        "new_hard_mean_percent": mean_percent(weighted_counts),
        "old_pooled": pooled_count(baseline_counts),
        "new_pooled": pooled_count(weighted_counts),
        "baseline_folds": [
            {"fold": fold_id, "correct": correct, "total": total}
            for fold_id, (correct, total) in enumerate(baseline_counts)
        ],
        "weighted_folds": report["folds"],
        "mining_fit_seconds": report["mining_fit_seconds"],
        "weighted_fit_seconds": report["weighted_fit_seconds"],
        "mining_optimizer_steps": report["mining_optimizer_steps"],
        "comparison_optimizer_steps": report["comparison_optimizer_steps"],
        "comparison_step_budget_matches_unweighted_recipe": True,
        "gate_seed": GATE_SEED,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_opened": False,
        "oof_logits": report["oof_logits"],
    }
    return gate, oof


def saved_arm_a_oof(emb: np.ndarray, seed: int, folds: Sequence[np.ndarray]) -> np.ndarray:
    oof = np.zeros((emb.shape[0], 4), dtype=np.float32)
    for fold_id, held in enumerate(folds):
        path = RUNS / f"exp_b12/seed_{seed}/arm_embeddings_only_fold{fold_id}.npz"
        oof[held] = score_saved_attention(path, emb[held])
    return oof


def train_full_weighted(
    train: SplitData,
    seed: int,
    device,
    registry: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Full-data arm A. Question weights come from the unweighted fold OOF."""
    folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
    oof = saved_arm_a_oof(train.emb, seed, folds)
    correct = hits_of(np.argmax(oof, axis=1), train.labels)
    hard = np.asarray(train.hard, dtype=bool)
    question_weight = question_loss_weights(correct, hard)
    seeds = full_model_seeds(seed)
    state, stats = train_weighted_arm(train.emb, train.labels, hard, question_weight, seeds, device)
    path = EXP_DIR / f"seed_{seed}" / "arm_a_weighted_full.npz"
    digest = save_npz(path, state)
    registry.append({
        "arm": "embeddings_only_weighted",
        "seed": int(seed),
        "member": "full",
        "derived": seeds,
        "file": rel(path),
        "sha256": digest,
        "role": "full train arm A; weights from the unweighted five-fold OOF of this seed",
    })
    print(
        f"  [seed {seed}] full weighted arm A "
        f"({float(stats['fit_seconds']):.1f}s, weight sum {float(question_weight.sum()):.1f})"
    )
    return state


def run_validation_confirmation(
    train: SplitData,
    device,
    seed42_oof: np.ndarray,
    registry: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Three-seed system comparison. Called only after the internal gate passes."""
    from experiments.exp_b07_cross_attn import standardize_apply
    from experiments.exp_b15_tree import fit_tree, tree_logits
    from experiments.exp_b22_g05_r5 import (
        HISTORICAL_ALL,
        HISTORICAL_HARD,
        HISTORICAL_WEIGHTS,
        fit_three_weights,
        fold_svd_blocks,
        official_score,
        retention_decision,
        score_choice,
        system_choice,
        write_official_predictions,
    )
    from experiments.multiseed import reasoning_struct

    print("\n--- Three-seed retrain (validation still closed) ---")
    folds = kfold_indices(len(train.bank), N_FOLDS, FOLD_PARTITION_SEED)
    weighted_oof = {GATE_SEED: seed42_oof}
    for seed in SEEDS:
        if seed == GATE_SEED:
            continue
        _report, oof = fit_nested_weighted_oof(train, seed, device, registry)
        weighted_oof[seed] = oof
    full_states = {seed: train_full_weighted(train, seed, device, registry) for seed in SEEDS}

    print("\n--- Validation confirmation ---")
    val = SplitData("validation")
    hard = np.asarray(val.hard, dtype=bool)
    val_weighted = {
        seed: CrossCandidateAttentionNet(full_states[seed]).forward_numpy(val.emb)
        for seed in SEEDS
    }
    struct16, _, _ = reasoning_struct(train)
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    saved_tree = np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")
    oof_a = {seed: saved_arm_a_oof(train.emb, seed, folds) for seed in SEEDS}
    oof_b = {seed: np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
    oof_tree = {seed: np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
    for fold_id, held in enumerate(folds):
        fit_rows = complement_rows(folds, [fold_id])
        for seed in SEEDS:
            weights_b = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_plus_g01_g03_fold{fold_id}.npz"))
            features_b = np.concatenate(
                [train.emb[held], standardize_apply(struct16[held], weights_b["struct_mean"], weights_b["struct_std"])],
                axis=-1,
            )
            oof_b[seed][held] = CrossCandidateAttentionNet(weights_b).forward_numpy(features_b)
        svd_fit, svd_held = fold_svd_blocks(train.emb[fit_rows], train.emb[held])
        fit_x = np.concatenate([struct16[fit_rows], svd_fit], axis=-1)
        held_x = np.concatenate([struct16[held], svd_held], axis=-1)
        for seed in SEEDS:
            tree, _stats = fit_tree("M04a", fit_x, train.labels[fit_rows], seed)
            oof_tree[seed][held] = tree_logits("M04a", tree, held_x).astype(np.float32)

    train_hard = np.asarray(train.hard, dtype=bool)
    paired = {}
    hard_deltas = []
    all_deltas = []
    for seed in SEEDS:
        weights_old = fit_three_weights(oof_a[seed], oof_b[seed], oof_tree[seed], train.labels, train_hard)
        weights_new = fit_three_weights(
            weighted_oof[seed], oof_b[seed], oof_tree[seed], train.labels, train_hard,
        )
        old_views = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], saved_tree[f"seed_{seed}"]]
        new_views = [val_weighted[seed], logits_b[f"seed_{seed}_full"], saved_tree[f"seed_{seed}"]]
        old_scored = score_choice(system_choice(weights_old, old_views, val.bank), val.labels, hard)
        new_scored = score_choice(system_choice(weights_new, new_views, val.bank), val.labels, hard)
        delta_hard = int(new_scored["val_hard"]["correct"] - old_scored["val_hard"]["correct"])
        delta_all = int(new_scored["val_all"]["correct"] - old_scored["val_all"]["correct"])
        hard_deltas.append(delta_hard)
        all_deltas.append(delta_all)
        paired[f"seed_{seed}"] = {
            "old_system": {
                "val_all": int(old_scored["val_all"]["correct"]),
                "val_hard": int(old_scored["val_hard"]["correct"]),
            },
            "new_system": {
                "val_all": int(new_scored["val_all"]["correct"]),
                "val_hard": int(new_scored["val_hard"]["correct"]),
            },
            "old_weights": [float(value) for value in weights_old],
            "new_weights": [float(value) for value in weights_new],
            "delta_hard": delta_hard,
            "delta_all": delta_all,
        }
        print(f"  seed {seed}: hard {delta_hard:+d}, all {delta_all:+d}")

    mean_old_a = np.mean([oof_a[seed] for seed in SEEDS], axis=0)
    mean_new_a = np.mean([weighted_oof[seed] for seed in SEEDS], axis=0)
    mean_b = np.mean([oof_b[seed] for seed in SEEDS], axis=0)
    ensemble_old = fit_three_weights(mean_old_a, mean_b, oof_tree[GATE_SEED], train.labels, train_hard)
    ensemble_new = fit_three_weights(mean_new_a, mean_b, oof_tree[GATE_SEED], train.labels, train_hard)
    old_ensemble_views = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], saved_tree["seed_42"]]
    new_ensemble_views = [
        np.mean([val_weighted[seed] for seed in SEEDS], axis=0),
        logits_b["seed_ensemble"],
        saved_tree["seed_42"],
    ]
    historical = score_choice(system_choice(HISTORICAL_WEIGHTS, old_ensemble_views, val.bank), val.labels, hard)
    strict_old = score_choice(system_choice(ensemble_old, old_ensemble_views, val.bank), val.labels, hard)
    new_choice = system_choice(ensemble_new, new_ensemble_views, val.bank)
    new_ensemble = score_choice(new_choice, val.labels, hard)
    if historical["val_hard"]["correct"] != HISTORICAL_HARD or historical["val_all"]["correct"] != HISTORICAL_ALL:
        raise RuntimeError("historical ensemble did not reproduce 300/355 and 743/798")
    if strict_old["val_hard"]["correct"] != 295 or strict_old["val_all"]["correct"] != 738:
        raise RuntimeError("refit strict three-view ensemble did not reproduce the B20 295/738 control")
    decision = retention_decision(
        hard_deltas, all_deltas, int(new_ensemble["val_hard"]["correct"]), int(new_ensemble["val_all"]["correct"]),
    )
    pred_path = EXP_DIR / "EXP-B23_weighted_arm_a_predictions.json"
    score_path = EXP_DIR / "EXP-B23_weighted_arm_a_score.json"
    write_official_predictions(pred_path, val.bank, new_choice)
    official = official_score(pred_path, score_path)
    if int(official["correct"]) != int(new_ensemble["val_all"]["correct"]):
        raise RuntimeError("official validation score does not match the in-process all-question count")
    return {
        "full_model_weight_source": (
            "Each full-model question weight comes from that seed's saved unweighted "
            "five-fold arm A. The fold model that labeled a question did not train on it."
        ),
        "paired_seed_evaluation": paired,
        "ensemble_old_weights": [float(value) for value in ensemble_old],
        "ensemble_new_weights": [float(value) for value in ensemble_new],
        "historical_reproduction": {
            "val_all": int(historical["val_all"]["correct"]),
            "val_hard": int(historical["val_hard"]["correct"]),
        },
        "strict_old_reproduction": {
            "val_all": int(strict_old["val_all"]["correct"]),
            "val_hard": int(strict_old["val_hard"]["correct"]),
        },
        "new_ensemble": {
            "val_all": int(new_ensemble["val_all"]["correct"]),
            "val_hard": int(new_ensemble["val_hard"]["correct"]),
        },
        "retention_gate": decision,
        "official_score": official,
        "validation_selection_configs": 9,
        "candidate_submission_updated": False,
    }


def run_exp_b23() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B23: nested hard-example loss weights for arm A")
    print("=" * 78)
    started = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rules = load_b23_rules()
    train = SplitData("train")
    device = configure_determinism()
    registry: List[Dict[str, Any]] = []
    gate, seed42_oof = run_internal_gate(train, device, rules["minimum_percentage_points"], registry)
    environment = environment_record(device)
    environment["interpreter"] = sys.executable
    audit: Dict[str, Any] = {
        "experiment": "EXP-B23",
        "description": "Arm A loss reweighting from nested OOF mining, with a train-internal gate",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment,
        "model_id": "MOD-M03",
        "sampling_loss_weights": {
            "training_prediction_incorrect": WEIGHT_INCORRECT,
            "training_prediction_correct_and_not_hard": WEIGHT_CORRECT_NOT_HARD,
            "other": WEIGHT_OTHER,
        },
        "internal_gate": gate,
        "validation_queries": 0,
        "test_references_opened": False,
        "written_to_model_bundle": False,
        "candidate_submission_updated": False,
    }
    register_seeds("EXP-B23", registry)
    if gate["passed"]:
        audit["verdict"] = "PENDING_VALIDATION"
        audit["validation_confirmation"] = None
        audit["execution_seconds"] = time.perf_counter() - started
        AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        confirmation = run_validation_confirmation(train, device, seed42_oof, registry)
        audit["validation_confirmation"] = confirmation
        audit["validation_queries"] = int(confirmation["validation_selection_configs"])
        audit["verdict"] = confirmation["retention_gate"]["verdict"]
        audit["candidate_submission_updated"] = False
        register_seeds("EXP-B23", registry)
    else:
        audit["verdict"] = "STOPPED_INTERNAL_GATE"
        audit["validation_confirmation"] = None
        print("Internal gate did not pass. Validation stays closed.")
    audit["execution_seconds"] = time.perf_counter() - started
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"VERDICT: {audit['verdict']}")
    print(f"Audit written to {AUDIT_PATH}")
    return audit


if __name__ == "__main__":
    run_exp_b23()
