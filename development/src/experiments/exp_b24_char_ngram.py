"""EXP-B24: character-hash TF-IDF logistic view, gated on train-internal folds.

The internal comparison rebuilds every base model that produces fusion targets
so that an outer held-out question is absent from those models and from the
inner OOF used to fit fusion weights. Globally precomputed OOF is not reused
for this gate. Validation is opened only after the held-out hard mean rises by
the pre-registered 1 percentage point. Test references are not opened.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
OFFICIAL_CODE = WORKSPACE / "DASE7506-MP2" / "code"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

from experiments.exp_b07_cross_attn import kfold_indices, standardize_apply, standardize_fit
from experiments.exp_b15_tree import fit_tree, tree_logits
from experiments.exp_b22_g05_r5 import (
    HISTORICAL_ALL,
    HISTORICAL_HARD,
    HISTORICAL_WEIGHTS,
    TEMPERATURE,
    fold_svd_blocks,
    official_score,
    retention_decision,
    score_choice,
    system_choice,
    write_official_predictions,
)
from experiments.multiseed import (
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    configure_determinism,
    environment_record,
    reasoning_struct,
    train_model,
)
from features.char_hash import (
    DIMENSIONS,
    NGRAMS,
    candidate_texts,
    count_candidates,
    normalize_candidate_text,
)
from models.cross_attn_net import CrossCandidateAttentionNet
from models.linear_view import TEXT_LOGISTIC_PARAMS, TextLogisticView

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
GATE_SEED = 42
EXP_DIR = RUNS / "exp_b24"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b24_char_ngram_audit.json"
CONTRACT_PATH = RUNS / "audit_logs" / "private_configuration_omitted.json"


def nested_attention_seeds(outer_fold: int, inner_fold: int) -> Dict[str, int]:
    """Pre-registered seed stream for a model that sees neither outer nor inner fold."""
    warmup = 100000 + 100 * int(outer_fold) + int(inner_fold)
    return {"warmup": warmup, "finetune": warmup + 1000}


def internal_gain_percentage_points(
    old_folds: Sequence[Tuple[int, int]],
    new_folds: Sequence[Tuple[int, int]],
) -> float:
    """Unweighted mean of fold hard-accuracy percentages, new minus old."""
    def mean_percent(folds: Sequence[Tuple[int, int]]) -> float:
        return float(np.mean([100.0 * correct / total for correct, total in folds]))

    return mean_percent(new_folds) - mean_percent(old_folds)


def question_candidates(rows: np.ndarray) -> np.ndarray:
    return (np.asarray(rows, dtype=np.int64)[:, None] * 4 + np.arange(4)).ravel()


def slice_counts(counts: Sequence[Mapping[int, int]], rows: np.ndarray) -> List[Mapping[int, int]]:
    index = question_candidates(rows)
    return [counts[int(i)] for i in index]


def fit_text_view(counts: Sequence[Mapping[int, int]], labels: np.ndarray, hard: np.ndarray, rows: np.ndarray) -> TextLogisticView:
    hard_rows = rows[np.asarray(hard, dtype=bool)[rows]]
    if hard_rows.size == 0:
        raise RuntimeError("text-view fit rows contain no hard questions")
    view = TextLogisticView()
    view.fit(slice_counts(counts, rows), slice_counts(counts, hard_rows), labels[hard_rows].reshape(-1))
    return view


def text_logits(view: TextLogisticView, counts: Sequence[Mapping[int, int]], rows: np.ndarray) -> np.ndarray:
    scores = view.decision_function(slice_counts(counts, rows))
    return scores.reshape(len(rows), 4)


def score_saved_attention(path: Path, features: np.ndarray) -> np.ndarray:
    weights = dict(np.load(path))
    return CrossCandidateAttentionNet(weights).forward_numpy(features)


def train_attention(features: np.ndarray, labels: np.ndarray, hard: np.ndarray, seeds: Dict[str, int], device: torch.device):
    local_hard = np.flatnonzero(np.asarray(hard, dtype=bool))
    if local_hard.size == 0:
        raise RuntimeError("attention fit rows contain no hard questions")
    weights, stats = train_model(
        features, labels, features[local_hard], labels[local_hard], seeds, device,
    )
    return weights, float(stats["fit_seconds"])


def hard_selection_count(choice: Sequence[int], labels: np.ndarray, hard: np.ndarray) -> Tuple[int, int]:
    chosen = np.asarray(choice, dtype=int)
    hits = labels[np.arange(labels.shape[0]), chosen]
    mask = np.asarray(hard, dtype=bool)
    return int(hits[mask].sum()), int(mask.sum())


def load_b24_contract() -> Dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    text = payload["text_view_contract"]
    if tuple(text["ngrams"]) != NGRAMS or int(text["dimensions"]) != DIMENSIONS:
        raise RuntimeError("character-hash constants do not match text_view_contract")
    if float(text["C"]) != TEXT_LOGISTIC_PARAMS["C"] or text["solver"] != TEXT_LOGISTIC_PARAMS["solver"]:
        raise RuntimeError("text logistic constants do not match text_view_contract")
    if int(text["max_iter"]) != TEXT_LOGISTIC_PARAMS["max_iter"]:
        raise RuntimeError("text logistic iteration limit does not match text_view_contract")
    minimum = float(payload["followup_batches"]["EXP-B24"]["train_internal_min_heldout_hard_gain_percentage_points"])
    return {"minimum_percentage_points": minimum, "text_view_contract": text}


def run_internal_gate(train: SplitData, counts: Sequence[Mapping[int, int]], device: torch.device, minimum: float) -> Dict[str, Any]:
    print("\n--- Internal five-fold gate (seed 42, nested OOF, validation closed) ---")
    struct16, _, _ = reasoning_struct(train)
    folds = kfold_indices(len(train.bank), 5, FOLD_PARTITION_SEED)
    hard = np.asarray(train.hard, dtype=bool)
    labels = train.labels
    started = time.perf_counter()
    attention_seconds = 0.0
    fold_rows = []
    old_counts: List[Tuple[int, int]] = []
    new_counts: List[Tuple[int, int]] = []
    text_only_counts: List[Tuple[int, int]] = []

    for outer, held in enumerate(folds):
        outer_rows = np.concatenate([folds[other] for other in range(5) if other != outer])
        held_emb = train.emb[held]
        arm_a_path = RUNS / f"exp_b12/seed_{GATE_SEED}/arm_embeddings_only_fold{outer}.npz"
        arm_b_path = RUNS / f"exp_b12/seed_{GATE_SEED}/arm_embeddings_plus_g01_g03_fold{outer}.npz"
        held_a = score_saved_attention(arm_a_path, held_emb)
        arm_b_weights = dict(np.load(arm_b_path))
        held_b_features = np.concatenate(
            [held_emb, standardize_apply(struct16[held], arm_b_weights["struct_mean"], arm_b_weights["struct_std"])],
            axis=-1,
        )
        held_b = CrossCandidateAttentionNet(arm_b_weights).forward_numpy(held_b_features)

        svd_fit, svd_held = fold_svd_blocks(train.emb[outer_rows], held_emb)
        tree, _ = fit_tree(
            "M04a",
            np.concatenate([struct16[outer_rows], svd_fit], axis=-1),
            labels[outer_rows],
            GATE_SEED,
        )
        held_tree = tree_logits("M04a", tree, np.concatenate([struct16[held], svd_held], axis=-1))
        outer_text = fit_text_view(counts, labels, hard, outer_rows)
        held_text = text_logits(outer_text, counts, held)

        oof_a = np.zeros((len(train.bank), 4), dtype=np.float32)
        oof_b = np.zeros_like(oof_a)
        oof_tree = np.zeros_like(oof_a)
        oof_text = np.zeros_like(oof_a)
        for inner, inner_held in enumerate(folds):
            if inner == outer:
                continue
            fit_rows = np.concatenate([folds[other] for other in range(5) if other not in (outer, inner)])
            seeds = nested_attention_seeds(outer, inner)
            weights_a, seconds_a = train_attention(train.emb[fit_rows], labels[fit_rows], hard[fit_rows], seeds, device)
            attention_seconds += seconds_a
            oof_a[inner_held] = CrossCandidateAttentionNet(weights_a).forward_numpy(train.emb[inner_held])

            mean, std = standardize_fit(struct16[fit_rows])
            features_b = np.concatenate(
                [train.emb[fit_rows], standardize_apply(struct16[fit_rows], mean, std)], axis=-1,
            )
            weights_b, seconds_b = train_attention(features_b, labels[fit_rows], hard[fit_rows], seeds, device)
            attention_seconds += seconds_b
            held_b_inner = np.concatenate(
                [train.emb[inner_held], standardize_apply(struct16[inner_held], mean, std)], axis=-1,
            )
            oof_b[inner_held] = CrossCandidateAttentionNet(weights_b).forward_numpy(held_b_inner)

            inner_svd_fit, inner_svd_held = fold_svd_blocks(train.emb[fit_rows], train.emb[inner_held])
            inner_tree, _ = fit_tree(
                "M04a",
                np.concatenate([struct16[fit_rows], inner_svd_fit], axis=-1),
                labels[fit_rows],
                GATE_SEED,
            )
            oof_tree[inner_held] = tree_logits(
                "M04a", inner_tree, np.concatenate([struct16[inner_held], inner_svd_held], axis=-1),
            ).astype(np.float32)
            inner_text = fit_text_view(counts, labels, hard, fit_rows)
            oof_text[inner_held] = text_logits(inner_text, counts, inner_held).astype(np.float32)
            print(f"  outer {outer} inner {inner} attention {seconds_a + seconds_b:.1f}s")

        weight_rows = np.zeros(len(train.bank), dtype=bool)
        weight_rows[outer_rows] = True
        weight_rows &= hard
        weights_old = _fit_views([oof_a, oof_b, oof_tree], labels, weight_rows)
        weights_new = _fit_views([oof_a, oof_b, oof_tree, oof_text], labels, weight_rows)
        bank_held = [train.bank[int(index)] for index in held]
        labels_held = labels[held]
        hard_held = hard[held]
        old_choice = system_choice(weights_old, [held_a, held_b, held_tree], bank_held)
        new_choice = system_choice(weights_new, [held_a, held_b, held_tree, held_text], bank_held)
        text_choice = np.argmax(held_text, axis=1)
        old_count = hard_selection_count(old_choice, labels_held, hard_held)
        new_count = hard_selection_count(new_choice, labels_held, hard_held)
        text_count = hard_selection_count(text_choice, labels_held, hard_held)
        old_counts.append(old_count)
        new_counts.append(new_count)
        text_only_counts.append(text_count)
        fold_rows.append({
            "outer_fold": outer,
            "held_questions": int(len(held)),
            "old_hard_correct": old_count[0],
            "new_hard_correct": new_count[0],
            "text_only_hard_correct": text_count[0],
            "hard_total": old_count[1],
            "weights_old": weights_old.tolist(),
            "weights_new": weights_new.tolist(),
        })
        print(
            f"Outer {outer}: old {old_count[0]}/{old_count[1]} new {new_count[0]}/{new_count[1]} "
            f"text-only {text_count[0]}/{text_count[1]}"
        )

    gain = internal_gain_percentage_points(old_counts, new_counts)
    passed = bool(gain >= minimum)
    print(f"Internal held-out hard mean gain: {gain:+.4f} percentage points (minimum {minimum:.1f})")
    return {
        "passed": passed,
        "gain_percentage_points": gain,
        "minimum_percentage_points": minimum,
        "folds": fold_rows,
        "old_hard_mean_percent": _mean_percent(old_counts),
        "new_hard_mean_percent": _mean_percent(new_counts),
        "text_only_hard_mean_percent": _mean_percent(text_only_counts),
        "attention_fit_seconds": attention_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "gate_seed": GATE_SEED,
        "nested_seed_formula": "warmup = 100000 + 100*outer + inner; finetune = warmup + 1000",
        "validation_opened": False,
    }


def _mean_percent(counts: Sequence[Tuple[int, int]]) -> float:
    return float(np.mean([100.0 * correct / total for correct, total in counts]))


def _fit_views(views: Sequence[np.ndarray], labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from postprocessing.soft_stacking import fit_convex_combination_weights, standardize_within_problem
    return fit_convex_combination_weights(
        [standardize_within_problem(view) for view in views], labels, subset_mask=mask,
    )


def run_validation_confirmation(train: SplitData, counts: Sequence[Mapping[int, int]]) -> Dict[str, Any]:
    """Three-seed validation pairing. Called only after the internal gate passes."""
    print("\n--- Validation confirmation (internal gate passed) ---")
    val = SplitData("validation")
    hard = np.asarray(val.hard, dtype=bool)
    val_counts = count_candidates(candidate_texts(val.bank))
    full_view = fit_text_view(counts, train.labels, np.asarray(train.hard), np.arange(len(train.bank)))
    val_text = text_logits(full_view, val_counts, np.arange(len(val.bank)))
    arrays = full_view.export_arrays()
    asset_path = EXP_DIR / "text_view.npz"
    np.savez(asset_path, **{key: value for key, value in arrays.items()})

    folds = kfold_indices(len(train.bank), 5, FOLD_PARTITION_SEED)
    oof_text = np.zeros((len(train.bank), 4), dtype=np.float32)
    for fold_id, held in enumerate(folds):
        fit_rows = np.concatenate([folds[other] for other in range(5) if other != fold_id])
        view = fit_text_view(counts, train.labels, np.asarray(train.hard), fit_rows)
        oof_text[held] = text_logits(view, counts, held).astype(np.float32)

    from experiments.exp_b22_g05_r5 import fit_three_weights
    struct16, _, _ = reasoning_struct(train)
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    saved_tree = np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")
    oof_a = {seed: np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
    oof_b = {seed: np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
    oof_tree = {seed: np.zeros((len(train.bank), 4), dtype=np.float32) for seed in SEEDS}
    for fold_id, held in enumerate(folds):
        fit_rows = np.concatenate([folds[other] for other in range(5) if other != fold_id])
        for seed in SEEDS:
            oof_a[seed][held] = score_saved_attention(
                RUNS / f"exp_b12/seed_{seed}/arm_embeddings_only_fold{fold_id}.npz", train.emb[held],
            )
            weights_b = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_plus_g01_g03_fold{fold_id}.npz"))
            features_b = np.concatenate(
                [train.emb[held], standardize_apply(struct16[held], weights_b["struct_mean"], weights_b["struct_std"])],
                axis=-1,
            )
            oof_b[seed][held] = CrossCandidateAttentionNet(weights_b).forward_numpy(features_b)
        svd_fit, svd_held = fold_svd_blocks(train.emb[fit_rows], train.emb[held])
        for seed in SEEDS:
            tree, _ = fit_tree(
                "M04a", np.concatenate([struct16[fit_rows], svd_fit], axis=-1), train.labels[fit_rows], seed,
            )
            oof_tree[seed][held] = tree_logits(
                "M04a", tree, np.concatenate([struct16[held], svd_held], axis=-1),
            ).astype(np.float32)

    train_hard = np.asarray(train.hard, dtype=bool)
    paired = {}
    hard_deltas = []
    all_deltas = []
    for seed in SEEDS:
        weights_old = fit_three_weights(oof_a[seed], oof_b[seed], oof_tree[seed], train.labels, train_hard)
        weights_new = _fit_views([oof_a[seed], oof_b[seed], oof_tree[seed], oof_text], train.labels, train_hard)
        old_views = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], saved_tree[f"seed_{seed}"]]
        new_views = old_views + [val_text]
        old_scored = score_choice(system_choice(weights_old, old_views, val.bank), val.labels, hard)
        new_scored = score_choice(system_choice(weights_new, new_views, val.bank), val.labels, hard)
        delta_hard = new_scored["val_hard"]["correct"] - old_scored["val_hard"]["correct"]
        delta_all = new_scored["val_all"]["correct"] - old_scored["val_all"]["correct"]
        hard_deltas.append(delta_hard)
        all_deltas.append(delta_all)
        paired[f"seed_{seed}"] = {
            "old_system": {"val_all": old_scored["val_all"]["correct"], "val_hard": old_scored["val_hard"]["correct"]},
            "new_system": {"val_all": new_scored["val_all"]["correct"], "val_hard": new_scored["val_hard"]["correct"]},
            "delta_hard": delta_hard,
            "delta_all": delta_all,
        }
        print(f"  seed {seed}: hard {delta_hard:+d}, all {delta_all:+d}")

    mean_a = np.mean([oof_a[seed] for seed in SEEDS], axis=0)
    mean_b = np.mean([oof_b[seed] for seed in SEEDS], axis=0)
    ensemble_old = fit_three_weights(mean_a, mean_b, oof_tree[42], train.labels, train_hard)
    ensemble_new = _fit_views([mean_a, mean_b, oof_tree[42], oof_text], train.labels, train_hard)
    old_ensemble_views = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], saved_tree["seed_42"]]
    historical = score_choice(system_choice(HISTORICAL_WEIGHTS, old_ensemble_views, val.bank), val.labels, hard)
    strict_old = score_choice(system_choice(ensemble_old, old_ensemble_views, val.bank), val.labels, hard)
    new_ensemble_choice = system_choice(ensemble_new, old_ensemble_views + [val_text], val.bank)
    new_ensemble = score_choice(new_ensemble_choice, val.labels, hard)
    if historical["val_hard"]["correct"] != HISTORICAL_HARD or historical["val_all"]["correct"] != HISTORICAL_ALL:
        raise RuntimeError("historical ensemble did not reproduce 300/355 and 743/798 before judging the text view")
    if strict_old["val_hard"]["correct"] != 295 or strict_old["val_all"]["correct"] != 738:
        raise RuntimeError("refit strict three-view ensemble did not reproduce the B20 295/738 control")
    decision = retention_decision(
        hard_deltas, all_deltas, new_ensemble["val_hard"]["correct"], new_ensemble["val_all"]["correct"],
    )
    pred_path = EXP_DIR / "EXP-B24_4view_predictions.json"
    score_path = EXP_DIR / "EXP-B24_4view_score.json"
    write_official_predictions(pred_path, val.bank, new_ensemble_choice)
    official = official_score(pred_path, score_path)
    asset_bytes = int(sum(np.asarray(value).nbytes for value in arrays.values()))
    return {
        "paired_seed_evaluation": paired,
        "ensemble_old_weights": ensemble_old.tolist(),
        "ensemble_new_weights": ensemble_new.tolist(),
        "historical_reproduction": {
            "val_all": historical["val_all"]["correct"],
            "val_hard": historical["val_hard"]["correct"],
        },
        "strict_old_reproduction": {
            "val_all": strict_old["val_all"]["correct"],
            "val_hard": strict_old["val_hard"]["correct"],
        },
        "new_ensemble": {
            "val_all": new_ensemble["val_all"]["correct"],
            "val_hard": new_ensemble["val_hard"]["correct"],
        },
        "retention_gate": decision,
        "official_score": official,
        "text_asset_bytes": asset_bytes,
        "text_asset": str(asset_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "validation_selection_configs": 9,
    }


def run_exp_b24() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B24: character-hash TF-IDF logistic view")
    print("=" * 78)
    started = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rules = load_b24_contract()
    train = SplitData("train")
    print("\n--- Hashing candidate text ---")
    hash_started = time.perf_counter()
    counts = count_candidates(candidate_texts(train.bank))
    print(f"Hashed {len(counts)} train candidates in {time.perf_counter() - hash_started:.1f}s")
    device = configure_determinism()
    gate = run_internal_gate(train, counts, device, rules["minimum_percentage_points"])
    environment = environment_record(device)
    environment["interpreter"] = sys.executable
    audit: Dict[str, Any] = {
        "experiment": "EXP-B24",
        "description": "Character n-gram hashed TF-IDF logistic view with a train-internal fusion gate",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment,
        "model_id": "MOD-M07b",
        "temperature_if_validation_opened": TEMPERATURE,
        "normalization_check": {
            "total_speed": normalize_candidate_text("The TOTAL is 60 km/h."),
            "decimal_line": normalize_candidate_text("Wait  2.50\nmore"),
            "comma_and_fraction": normalize_candidate_text("1,000 apples and 1/4."),
        },
        "internal_gate": gate,
        "validation_queries": 0,
        "test_references_opened": False,
        "written_to_model_bundle": False,
    }
    if gate["passed"]:
        confirmation = run_validation_confirmation(train, counts)
        audit["validation_confirmation"] = confirmation
        audit["validation_queries"] = confirmation["validation_selection_configs"]
        audit["verdict"] = confirmation["retention_gate"]["verdict"]
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
    run_exp_b24()
