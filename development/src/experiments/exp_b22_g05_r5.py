"""EXP-B22: replace Tree R5 with a 150-d model that keeps SVD and adds FEAT-G05.

The deployed tree is HistGradientBoosting on 16 raw structured columns plus the
128-d FEAT-G04b block (64 SVD + 64 within-question centered), fit on train_all.
This batch appends the 6 FEAT-G05 columns (22 + 128 = 150) and refits only the
tree inside the B17/B20 three-view system. The B21 linear view is not added.

Rules, from private_document_omitted and private_configuration_omitted.json:
- Column names are read from the follow-up contract, not copied by hand.
- Structured columns stay on the raw scale used by EXP-B15 / the B20 strict OOF.
- SVD for the full model is the B15 train_all TruncatedSVD (random_state=42).
- Fold OOF refits SVD inside the training fold. Tree hyperparameters are unchanged.
- Fusion weights are fit on training OOF only. Temperature stays at the historical 0.5.
- Test references are not opened. submission.py and model_bundle are not written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
OFFICIAL_CODE = WORKSPACE / "DASE7506-MP2" / "code"

if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

import common  # type: ignore
from experiments.exp_b07_cross_attn import kfold_indices
from experiments.exp_b15_tree import HGB_PARAMS, fit_tree, svd_features, tree_logits
from experiments.multiseed import (
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    environment_record,
    reasoning_struct,
    register_seeds,
    rel,
)
from features.finegrained import FineGrainedVerificationFeatureExtractor
from models.cross_attn_net import CrossCandidateAttentionNet
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
)
from postprocessing.soft_stacking import (
    fit_convex_combination_weights,
    standardize_within_problem,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
TEMPERATURE = 0.5
MASK_PENALTY = 1e9
SVD_SEED = 42
OLD_DIM = 144
NEW_DIM = 150
HISTORICAL_WEIGHTS = np.array(
    [0.6519943983184848, 0.2773940600630942, 0.07061154161842091], dtype=np.float64
)
B20_STRICT_WEIGHTS = np.array(
    [0.6401443191116853, 0.2887782261683207, 0.07107745471999402], dtype=np.float64
)
HISTORICAL_HARD = 300
HISTORICAL_ALL = 743
B20_STRICT_HARD = 295
B20_STRICT_ALL = 738
PAIRED_HARD_GAIN = 3.55
ENSEMBLE_HARD_GAIN = 4

EXP_B22_DIR = RUNS / "exp_b22"
AUDIT_DIR = RUNS / "audit_logs"
CONTRACT_PATH = AUDIT_DIR / "private_configuration_omitted.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def load_followup_contract() -> Dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def accuracy_dict(hits: np.ndarray) -> Dict[str, Any]:
    total = int(hits.size)
    correct = int(np.asarray(hits).sum())
    return {"correct": correct, "total": total, "accuracy": 100.0 * correct / max(1, total)}


def score_choice(
    chosen: Sequence[int],
    labels: np.ndarray,
    hard: np.ndarray,
) -> Dict[str, Any]:
    idx = np.arange(labels.shape[0])
    hits = labels[idx, np.asarray(chosen, dtype=int)].astype(bool)
    easy = ~np.asarray(hard, dtype=bool)
    return {
        "val_all": accuracy_dict(hits),
        "val_easy": accuracy_dict(hits[easy]),
        "val_hard": accuracy_dict(hits[hard]),
    }


def decode_system(logits: np.ndarray, bank: Sequence[Dict[str, Any]]) -> List[int]:
    masked = apply_validity_mask(logits, bank, penalty=MASK_PENALTY)
    masked = apply_conflict_penalty(masked, bank, penalty=MASK_PENALTY)
    return SoftmaxClusterAggregator(temperature=TEMPERATURE).aggregate(bank, masked)


def blend(weights: Sequence[float], views: Sequence[np.ndarray]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    stacked = np.zeros(views[0].shape, dtype=np.float64)
    for weight, logits in zip(weights, views):
        stacked += weight * standardize_within_problem(logits)
    return stacked


def system_choice(weights: Sequence[float], views: Sequence[np.ndarray], bank: Sequence[Dict[str, Any]]) -> List[int]:
    return decode_system(blend(weights, views), bank)


def append_g05(split: SplitData, finegrained_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Raw 16-column reasoning block plus FEAT-G05 in contract order."""
    struct16, g01_names, g03_names = reasoning_struct(split)
    extractor = FineGrainedVerificationFeatureExtractor()
    raw = extractor.transform(split.bank, split.arrays)
    index = [extractor.FEATURE_NAMES.index(name) for name in finegrained_names]
    g05 = raw[:, :, index].astype(np.float32)
    names = list(g01_names) + list(g03_names) + list(finegrained_names)
    return struct16, np.concatenate([struct16, g05], axis=-1), names


def fold_svd_blocks(fit_emb: np.ndarray, held_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """128-d SVD block fit only on the training fold. Matches B15's transform."""
    svd = TruncatedSVD(n_components=64, random_state=SVD_SEED)
    svd.fit(fit_emb.reshape(-1, fit_emb.shape[-1]))

    def block(emb: np.ndarray) -> np.ndarray:
        proj = svd.transform(emb.reshape(-1, emb.shape[-1])).reshape(emb.shape[0], 4, 64)
        centered = proj - proj.mean(axis=1, keepdims=True)
        return np.concatenate([proj, centered], axis=-1).astype(np.float32)

    return block(fit_emb), block(held_emb)


def attention_held(
    train: SplitData,
    struct16: np.ndarray,
    seed: int,
    fold_id: int,
    held: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    held_emb = train.emb[held]
    weights_a = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_only_fold{fold_id}.npz"))
    logits_a = CrossCandidateAttentionNet(weights_a).forward_numpy(held_emb)
    weights_b = dict(np.load(RUNS / f"exp_b12/seed_{seed}/arm_embeddings_plus_g01_g03_fold{fold_id}.npz"))
    mean, std = weights_b["struct_mean"], weights_b["struct_std"]
    struct_norm = (struct16[held] - mean) / np.maximum(std, 1e-6)
    features_b = np.concatenate([held_emb, struct_norm], axis=-1)
    logits_b = CrossCandidateAttentionNet(weights_b).forward_numpy(features_b)
    return logits_a, logits_b


def fit_three_weights(
    arm_a: np.ndarray,
    arm_b: np.ndarray,
    tree: np.ndarray,
    labels: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    return fit_convex_combination_weights(
        [
            standardize_within_problem(arm_a),
            standardize_within_problem(arm_b),
            standardize_within_problem(tree),
        ],
        labels,
        subset_mask=hard_mask,
    )


def g05_importance(model, column_names: Sequence[str]) -> Dict[str, Any]:
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
    if importances.size != NEW_DIM:
        return {"available": False}
    total = float(importances.sum())
    named = {name: float(importances[i]) for i, name in enumerate(column_names)}
    g05_names = list(column_names[16:22])
    return {
        "available": True,
        "g05_share": float(importances[16:22].sum() / max(total, 1e-12)),
        "svd_share": float(importances[22:].sum() / max(total, 1e-12)),
        "structured16_share": float(importances[:16].sum() / max(total, 1e-12)),
        "g05_columns": {name: named[name] for name in g05_names},
    }


def retention_decision(
    paired_hard_deltas: Sequence[float],
    paired_all_deltas: Sequence[float],
    ensemble_hard: int,
    ensemble_all: int,
) -> Dict[str, Any]:
    """Pre-registered B20–B24 gate. Strict-control gains use the published B20 numbers."""
    hard_deltas = [float(value) for value in paired_hard_deltas]
    all_deltas = [float(value) for value in paired_all_deltas]
    mean_hard = float(np.mean(hard_deltas))
    mean_all = float(np.mean(all_deltas))
    positive = int(sum(value > 0 for value in hard_deltas))
    hist_hard_gain = int(ensemble_hard) - HISTORICAL_HARD
    hist_all_gain = int(ensemble_all) - HISTORICAL_ALL
    strict_hard_gain = int(ensemble_hard) - B20_STRICT_HARD
    strict_all_gain = int(ensemble_all) - B20_STRICT_ALL
    gates = {
        "rule1_paired_hard_mean_ge_3_55": {"value": mean_hard, "passed": bool(mean_hard >= PAIRED_HARD_GAIN)},
        "rule2_paired_hard_positive_ge_2": {"value": positive, "passed": bool(positive >= 2)},
        "rule3_paired_all_mean_ge_0": {"value": mean_all, "passed": bool(mean_all >= 0.0)},
        "rule4_ensemble_hard_vs_historical_ge_4": {
            "value": hist_hard_gain,
            "passed": bool(hist_hard_gain >= ENSEMBLE_HARD_GAIN),
        },
        "rule5_ensemble_all_vs_historical_ge_0": {"value": hist_all_gain, "passed": bool(hist_all_gain >= 0)},
        "rule6_ensemble_hard_vs_b20_strict_ge_4": {
            "value": strict_hard_gain,
            "passed": bool(strict_hard_gain >= ENSEMBLE_HARD_GAIN),
        },
        "rule7_ensemble_all_vs_b20_strict_ge_0": {"value": strict_all_gain, "passed": bool(strict_all_gain >= 0)},
    }
    retained = all(item["passed"] for item in gates.values())
    return {"gates": gates, "retained": retained, "verdict": "RETAINED" if retained else "REJECTED"}


def write_official_predictions(path: Path, bank: Sequence[Dict[str, Any]], chosen: Sequence[int]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "split": "validation",
        "ids": [row["id"] for row in bank],
        "selected": [int(index) for index in chosen],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def official_score(predictions: Path, output: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    subprocess.run(
        [
            sys.executable,
            str(OFFICIAL_CODE / "score_predictions.py"),
            "--split",
            "validation",
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def run_exp_b22() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B22: 150-d Tree R5 (SVD retained, FEAT-G05 appended)")
    print("=" * 78)
    started = time.perf_counter()
    EXP_B22_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    contract = load_followup_contract()["column_contract"]
    print("\n--- Step 1: Features (raw structured_22 + B15 SVD) ---")
    train = SplitData("train")
    val = SplitData("validation")
    hard = np.asarray(val.hard, dtype=bool)
    train_hard = np.asarray(train.hard, dtype=bool)
    feature_started = time.perf_counter()
    struct16_train, struct22_train, column_names = append_g05(train, contract["finegrained"])
    struct16_val, struct22_val, val_names = append_g05(val, contract["finegrained"])
    if column_names != contract["structured_22"] or val_names != contract["structured_22"]:
        raise RuntimeError("structured_22 column order does not match the follow-up contract")
    if column_names[:16] != contract["structured_16"]:
        raise RuntimeError("the first 16 columns are not the deployed R5 structured block")
    svd_train, svd_val, _svd = svd_features(train, val, seed=SVD_SEED)
    if struct22_train.shape[-1] != 22 or svd_train.shape[-1] != 128:
        raise RuntimeError(f"unexpected feature widths: {struct22_train.shape[-1]} + {svd_train.shape[-1]}")
    x_new_train = np.concatenate([struct22_train, svd_train], axis=-1)
    x_new_val = np.concatenate([struct22_val, svd_val], axis=-1)
    x_old_val = np.concatenate([struct16_val, svd_val], axis=-1)
    print(f"Features ready in {time.perf_counter() - feature_started:.2f}s; new input dim {x_new_train.shape[-1]}.")

    print("\n--- Step 2: Confirm the saved R5 sees these 144 columns ---")
    saved_logits = np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")
    with (RUNS / "exp_b15/seed_42/M04a_train_all_plus_svd.pkl").open("rb") as handle:
        old_model = pickle.load(handle)
    recomputed = tree_logits("M04a", old_model, x_old_val)
    logit_gap = float(np.max(np.abs(recomputed - saved_logits["seed_42"])))
    old_choice = np.argmax(saved_logits["seed_42"], axis=1)
    old_standalone = score_choice(old_choice, val.labels, hard)
    print(f"Saved-vs-recomputed logit max abs gap: {logit_gap:.3e}")
    print(
        f"Old R5 argmax seed 42: {old_standalone['val_all']['correct']}/798, "
        f"hard {old_standalone['val_hard']['correct']}/355"
    )
    if old_standalone["val_hard"]["correct"] != 268 or old_standalone["val_all"]["correct"] != 711:
        raise RuntimeError("old R5 argmax did not reproduce the B15 268/711 record")
    if logit_gap > 1e-4:
        raise RuntimeError(f"rebuilt 144-d features do not match the saved R5 logits (gap {logit_gap})")

    old_standalone_seeds = {}
    old_predictions = []
    for seed in SEEDS:
        choice = np.argmax(saved_logits[f"seed_{seed}"], axis=1)
        scored = score_choice(choice, val.labels, hard)
        old_standalone_seeds[str(seed)] = {
            "val_all": scored["val_all"]["correct"],
            "val_hard": scored["val_hard"]["correct"],
        }
        old_predictions.append(choice.tolist())
    old_prediction_identical = all(item == old_predictions[0] for item in old_predictions)

    print("\n--- Step 3: Fit three full 150-d trees on train_all ---")
    new_logits = {}
    new_standalone = {}
    registry = []
    importance = {}
    asset_files = []
    for seed in SEEDS:
        model, fit_seconds = fit_tree("M04a", x_new_train, train.labels, seed)
        logits = tree_logits("M04a", model, x_new_val)
        new_logits[seed] = logits
        choice = np.argmax(logits, axis=1)
        scored = score_choice(choice, val.labels, hard)
        path = EXP_B22_DIR / f"seed_{seed}" / "M04a_train_all_g05_svd.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        digest = sha256_file(path)
        asset_files.append({"file": rel(path), "sha256": digest, "bytes": path.stat().st_size, "seed": seed})
        registry.append({
            "arm": "M04a_train_all_g05_svd",
            "seed": seed,
            "member": "full",
            "derived": {"random_state": seed},
            "file": rel(path),
            "sha256": digest,
        })
        new_standalone[str(seed)] = {
            "val_all": scored["val_all"]["correct"],
            "val_hard": scored["val_hard"]["correct"],
            "fit_seconds": fit_seconds,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        if seed == 42:
            importance = g05_importance(model, column_names + [f"svd_{i}" for i in range(64)] + [f"svd_centered_{i}" for i in range(64)])
        print(
            f"  seed {seed}: argmax {scored['val_all']['correct']}/798, "
            f"hard {scored['val_hard']['correct']}/355, fit {fit_seconds:.2f}s, {path.stat().st_size} bytes"
        )
    new_prediction_identical = all(
        np.array_equal(np.argmax(new_logits[seed], axis=1), np.argmax(new_logits[SEEDS[0]], axis=1))
        for seed in SEEDS
    )
    np.savez(EXP_B22_DIR / "val_logits_new_r5.npz", **{f"seed_{seed}": new_logits[seed] for seed in SEEDS})
    register_seeds("EXP-B22", registry)

    print("\n--- Step 4: Strict fold OOF (shared fold SVD, raw columns, three tree seeds) ---")
    n_train = len(train.bank)
    folds = kfold_indices(n_train, 5, FOLD_PARTITION_SEED)
    oof_a = {seed: np.zeros((n_train, 4), dtype=np.float32) for seed in SEEDS}
    oof_b = {seed: np.zeros((n_train, 4), dtype=np.float32) for seed in SEEDS}
    oof_old = {seed: np.zeros((n_train, 4), dtype=np.float32) for seed in SEEDS}
    oof_new = {seed: np.zeros((n_train, 4), dtype=np.float32) for seed in SEEDS}
    oof_started = time.perf_counter()
    for fold_id, held in enumerate(folds):
        fit_rows = np.concatenate([folds[other] for other in range(5) if other != fold_id])
        for seed in SEEDS:
            logits_a, logits_b = attention_held(train, struct16_train, seed, fold_id, held)
            oof_a[seed][held] = logits_a
            oof_b[seed][held] = logits_b
        svd_fit, svd_held = fold_svd_blocks(train.emb[fit_rows], train.emb[held])
        old_fit = np.concatenate([struct16_train[fit_rows], svd_fit], axis=-1)
        old_held = np.concatenate([struct16_train[held], svd_held], axis=-1)
        new_fit = np.concatenate([struct22_train[fit_rows], svd_fit], axis=-1)
        new_held = np.concatenate([struct22_train[held], svd_held], axis=-1)
        for seed in SEEDS:
            old_fold, _ = fit_tree("M04a", old_fit, train.labels[fit_rows], seed)
            new_fold, _ = fit_tree("M04a", new_fit, train.labels[fit_rows], seed)
            oof_old[seed][held] = tree_logits("M04a", old_fold, old_held).astype(np.float32)
            oof_new[seed][held] = tree_logits("M04a", new_fold, new_held).astype(np.float32)
        print(f"  fold {fold_id} done at {time.perf_counter() - oof_started:.1f}s")
    oof_seconds = time.perf_counter() - oof_started
    print(f"OOF finished in {oof_seconds:.1f}s.")

    print("\n--- Step 5: Train-only fusion weights ---")
    paired_weights = {}
    for seed in SEEDS:
        old_w = fit_three_weights(oof_a[seed], oof_b[seed], oof_old[seed], train.labels, train_hard)
        new_w = fit_three_weights(oof_a[seed], oof_b[seed], oof_new[seed], train.labels, train_hard)
        paired_weights[str(seed)] = {"old_tree": old_w.tolist(), "new_tree": new_w.tolist()}
        print(f"  seed {seed} old {np.round(old_w, 4).tolist()} new {np.round(new_w, 4).tolist()}")

    mean_a = np.mean([oof_a[seed] for seed in SEEDS], axis=0)
    mean_b = np.mean([oof_b[seed] for seed in SEEDS], axis=0)
    ensemble_old_weights = fit_three_weights(mean_a, mean_b, oof_old[42], train.labels, train_hard)
    ensemble_new_weights = fit_three_weights(mean_a, mean_b, oof_new[42], train.labels, train_hard)
    weight_gap = float(np.max(np.abs(ensemble_old_weights - B20_STRICT_WEIGHTS)))
    print(f"Ensemble old weights {np.round(ensemble_old_weights, 4).tolist()} (gap vs B20 {weight_gap:.3e})")
    print(f"Ensemble new weights {np.round(ensemble_new_weights, 4).tolist()}")
    # SLSQP can stop a few thousandths away on this flat simplex. The published
    # B20 control is the validation score, checked after decoding below.

    print("\n--- Step 6: Validation systems (pre-registered, no temperature search) ---")
    logits_a = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    paired = {}
    hard_deltas = []
    all_deltas = []
    for seed in SEEDS:
        views_old = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], saved_logits[f"seed_{seed}"]]
        views_new = [logits_a[f"seed_{seed}_full"], logits_b[f"seed_{seed}_full"], new_logits[seed]]
        old_choice_seed = system_choice(paired_weights[str(seed)]["old_tree"], views_old, val.bank)
        new_choice_seed = system_choice(paired_weights[str(seed)]["new_tree"], views_new, val.bank)
        old_scored = score_choice(old_choice_seed, val.labels, hard)
        new_scored = score_choice(new_choice_seed, val.labels, hard)
        delta_hard = new_scored["val_hard"]["correct"] - old_scored["val_hard"]["correct"]
        delta_all = new_scored["val_all"]["correct"] - old_scored["val_all"]["correct"]
        hard_deltas.append(delta_hard)
        all_deltas.append(delta_all)
        paired[f"seed_{seed}"] = {
            "old_system": {"val_all": old_scored["val_all"]["correct"], "val_hard": old_scored["val_hard"]["correct"]},
            "new_system": {"val_all": new_scored["val_all"]["correct"], "val_hard": new_scored["val_hard"]["correct"]},
            "delta_hard": delta_hard,
            "delta_all": delta_all,
            "weights_old": paired_weights[str(seed)]["old_tree"],
            "weights_new": paired_weights[str(seed)]["new_tree"],
        }
        print(
            f"  seed {seed}: old {old_scored['val_all']['correct']}/{old_scored['val_hard']['correct']} -> "
            f"new {new_scored['val_all']['correct']}/{new_scored['val_hard']['correct']} "
            f"(hard {delta_hard:+d}, all {delta_all:+d})"
        )

    ensemble_views_old = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], saved_logits["seed_42"]]
    ensemble_views_new = [logits_a["seed_ensemble"], logits_b["seed_ensemble"], new_logits[42]]
    historical_choice = system_choice(HISTORICAL_WEIGHTS, ensemble_views_old, val.bank)
    strict_choice = system_choice(ensemble_old_weights, ensemble_views_old, val.bank)
    new_choice = system_choice(ensemble_new_weights, ensemble_views_new, val.bank)
    historical_scored = score_choice(historical_choice, val.labels, hard)
    strict_scored = score_choice(strict_choice, val.labels, hard)
    new_scored = score_choice(new_choice, val.labels, hard)
    print(
        f"Historical ensemble {historical_scored['val_all']['correct']}/{historical_scored['val_hard']['correct']} | "
        f"strict old {strict_scored['val_all']['correct']}/{strict_scored['val_hard']['correct']} | "
        f"new {new_scored['val_all']['correct']}/{new_scored['val_hard']['correct']}"
    )
    if historical_scored["val_hard"]["correct"] != HISTORICAL_HARD or historical_scored["val_all"]["correct"] != HISTORICAL_ALL:
        raise RuntimeError("historical three-view ensemble did not reproduce 300/355 and 743/798")
    if strict_scored["val_hard"]["correct"] != B20_STRICT_HARD or strict_scored["val_all"]["correct"] != B20_STRICT_ALL:
        raise RuntimeError("strict old ensemble did not reproduce the B20 295/738 control")

    decision = retention_decision(hard_deltas, all_deltas, new_scored["val_hard"]["correct"], new_scored["val_all"]["correct"])
    print(f"VERDICT: {decision['verdict']}")
    for name, gate in decision["gates"].items():
        print(f"  {name}: {gate['value']} ({'PASS' if gate['passed'] else 'FAIL'})")

    pred_path = EXP_B22_DIR / "EXP-B22_3view_new_r5_predictions.json"
    score_path = EXP_B22_DIR / "EXP-B22_3view_new_r5_score.json"
    write_official_predictions(pred_path, val.bank, new_choice)
    official = official_score(pred_path, score_path)
    if int(official["correct"]) != new_scored["val_all"]["correct"]:
        raise RuntimeError("official scorer disagreed with the label-matrix count on the new ensemble")

    tree_bytes = int(sum(item["bytes"] for item in asset_files))
    audit = {
        "experiment": "EXP-B22",
        "description": "Retain R5 SVD, append FEAT-G05, refit the three-view tree member",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_record(torch.device("cpu")),
        "model_id": "MOD-M04a",
        "hyperparameters": HGB_PARAMS,
        "fit_rows": "train_all",
        "feature_scale": "raw_structured_plus_svd",
        "input_dimensions": {"structured": 22, "svd": 128, "total": NEW_DIM, "old_total": OLD_DIM},
        "columns_structured_22": column_names,
        "svd": {"source": "exp_b15 svd_features", "random_state": SVD_SEED, "fold_local_for_oof": True},
        "b21_linear_view_included": False,
        "temperature": TEMPERATURE,
        "old_r5_feature_gap_max_abs": logit_gap,
        "standalone_argmax": {
            "old_r5": old_standalone_seeds,
            "new_r5": new_standalone,
            "old_predictions_identical": old_prediction_identical,
            "new_predictions_identical": new_prediction_identical,
            "note": "Identical predictions are required before calling the tree seed-insensitive. File hashes may still differ.",
        },
        "feature_importance_seed_42": importance,
        "oof": {
            "seconds": oof_seconds,
            "folds": 5,
            "fold_partition_seed": FOLD_PARTITION_SEED,
            "attention": "per-seed fold models; ensemble weights use the three-seed mean",
            "tree_member_for_ensemble_weights": 42,
            "weight_gap_vs_b20_strict_max_abs": weight_gap,
            "ensemble_old_weights": ensemble_old_weights.tolist(),
            "ensemble_new_weights": ensemble_new_weights.tolist(),
        },
        "paired_seed_evaluation": paired,
        "final_ensemble": {
            "definition": "three-seed mean of arm A and arm B, plus the seed-42 tree",
            "historical_old_tree": {
                "val_all": historical_scored["val_all"]["correct"],
                "val_hard": historical_scored["val_hard"]["correct"],
                "weights": HISTORICAL_WEIGHTS.tolist(),
            },
            "strict_old_tree": {
                "val_all": strict_scored["val_all"]["correct"],
                "val_hard": strict_scored["val_hard"]["correct"],
                "weights": ensemble_old_weights.tolist(),
            },
            "new_tree": {
                "val_all": new_scored["val_all"]["correct"],
                "val_hard": new_scored["val_hard"]["correct"],
                "weights": ensemble_new_weights.tolist(),
            },
        },
        "retention_gate": decision,
        "validation_configuration_ledger": {
            "pre_registered_selection_configs": 15,
            "executed_selection_configs": 15,
            "breakdown": {
                "old_tree_argmax_seeds": 3,
                "new_tree_argmax_seeds": 3,
                "paired_old_systems": 3,
                "paired_new_systems": 3,
                "ensembles": 3,
            },
            "official_score_calls": 1,
            "oof_weight_fits_train_only": 8,
            "reruns": 0,
            "temperature_or_hyperparameter_search_on_validation": 0,
        },
        "assets": {
            "files": asset_files,
            "new_tree_pickle_bytes_sum": tree_bytes,
            "deployed_old_tree_bytes": (RUNS / "exp_b15/seed_42/M04a_train_all_plus_svd.pkl").stat().st_size,
            "written_to_model_bundle": False,
            "resource_note": "Pickle bytes are the candidate tree assets. Full four-thread predict-plus-score wall clock stays with the frozen bundle and was not claimed here.",
        },
        "official_score": official,
        "execution_seconds": time.perf_counter() - started,
    }
    audit_path = AUDIT_DIR / "exp_b22_g05_r5_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Audit written to {audit_path}")
    print(f"Total {audit['execution_seconds']:.1f}s")
    return audit


if __name__ == "__main__":
    run_exp_b22()
