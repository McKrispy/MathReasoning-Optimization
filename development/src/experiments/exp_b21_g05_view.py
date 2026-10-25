"""EXP-B21: G05 Fine-Grained Symbolic Linear View Integration & Stacking Ablation.

Evaluates adding a 4th heterogeneous view:
- View 1: Arm A (pure geometric semantic cross-candidate attention)
- View 2: Arm B (symbolic reasoning + consensus cross-candidate attention)
- View 3: Tree R5 (HistGradientBoosting on 16 structured + 128 SVD features)
- View 4: MOD-M07a (Standardized L2 Logistic Regression on 22 structured columns:
          Consensus Core 4 + Symbolic 12 + Fine-grained G05 6)

Follows the EXP-B21 contract from private_configuration_omitted.json:
- Classifier: L2 LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
- Features: structured_22
- Fits on train_hard only, fold-local standardization on fit_hard
- Stacking weights fit on training set OOF with non-negative convex combination (sum=1)
- Evaluates against B20 confirmed baseline with pre-registered retention rules
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
OFFICIAL_CODE = WORKSPACE / "DASE7506-MP2" / "code"

if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

import common  # type: ignore
from common_utils import (
    build_label_matrix,
    get_hard_subset_mask,
    load_dataset,
    load_gold_references,
)
from experiments.exp_b07_cross_attn import kfold_indices
from experiments.exp_b15_tree import HGB_PARAMS
from experiments.multiseed import (
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    SplitData,
    environment_record,
    reasoning_struct,
)
from features.consensus import ConsensusFeatureExtractor
from features.finegrained import FineGrainedVerificationFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor
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
EXP_B21_DIR = RUNS / "exp_b21"
AUDIT_DIR = RUNS / "audit_logs"


def accuracy_dict(hits: np.ndarray) -> Dict[str, Any]:
    n = int(hits.size)
    c = int(hits.sum())
    return {"correct": c, "total": n, "accuracy": 100.0 * c / max(1, n)}


def evaluate_choices(
    chosen: Sequence[int],
    bank: Sequence[Dict[str, Any]],
    gold_answers: Sequence[str],
    is_hard: np.ndarray,
) -> Dict[str, Any]:
    hits = np.array(
        [common.parse_answer(bank[i]["candidates"][chosen[i]]["text"]) == gold_answers[i] for i in range(len(bank))],
        dtype=bool,
    )
    is_easy = ~is_hard
    return {
        "val_all": accuracy_dict(hits),
        "val_easy": accuracy_dict(hits[is_easy]),
        "val_hard": accuracy_dict(hits[is_hard]),
    }


def write_official_predictions(path: Path, bank: Sequence[Dict[str, Any]], chosen: Sequence[int]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "split": "validation",
        "ids": [row["id"] for row in bank],
        "selected": [int(idx) for idx in chosen],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def official_score(predictions: Path, output: Path) -> Dict[str, Any]:
    scorer = OFFICIAL_CODE / "score_predictions.py"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    subprocess.run(
        [sys.executable, str(scorer), "--split", "validation", "--predictions", str(predictions), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def run_exp_b21() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B21: G05 Fine-Grained Symbolic View Integration & Stacking Ablation")
    print("=" * 78)

    t0 = time.perf_counter()
    EXP_B21_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets
    print("\n--- Step 1: Loading Datasets ---")
    train = SplitData("train")
    val = SplitData("validation")
    gold_val = load_gold_references("validation")
    gold_answers = [common.parse_answer(q["answer"]) for q in gold_val]

    n_train = len(train.bank)
    n_val = len(val.bank)
    is_hard = np.array(val.hard, dtype=bool)
    is_easy = ~is_hard
    train_hard_mask = np.array(train.hard, dtype=bool)

    print(f"Dataset: train={n_train} questions, validation={n_val} questions ({is_easy.sum()} easy, {is_hard.sum()} hard).")

    # 2. Extract structured_22 features (Consensus Core 4 + Symbolic 12 + G05 6)
    print("\n--- Step 2: Extracting structured_22 Feature Families ---")
    t_feat = time.perf_counter()

    ext_g01 = ConsensusFeatureExtractor()
    ext_g03 = SymbolicVerificationFeatureExtractor()
    ext_g05 = FineGrainedVerificationFeatureExtractor()

    contract_path = AUDIT_DIR / "private_configuration_omitted.json"
    with contract_path.open("r", encoding="utf-8") as f:
        contract_data = json.load(f)
    col_contract = contract_data["column_contract"]

    core_cols = col_contract["consensus_core"]
    symbolic_cols = col_contract["symbolic"]
    g05_cols = col_contract["finegrained"]
    expected_22 = col_contract["structured_22"]

    core_idx = [ext_g01.FEATURE_NAMES.index(c) for c in core_cols]
    sym_idx = [ext_g03.FEATURE_NAMES.index(c) for c in symbolic_cols]

    # Train features
    tr_g01 = ext_g01.transform(train.bank, train.arrays)[:, :, core_idx].astype(np.float32)
    tr_g03 = ext_g03.transform(train.bank, train.arrays)[:, :, sym_idx].astype(np.float32)
    tr_g05 = ext_g05.transform(train.bank, train.arrays).astype(np.float32)
    struct_22_train = np.concatenate([tr_g01, tr_g03, tr_g05], axis=-1)

    # Validation features
    va_g01 = ext_g01.transform(val.bank, val.arrays)[:, :, core_idx].astype(np.float32)
    va_g03 = ext_g03.transform(val.bank, val.arrays)[:, :, sym_idx].astype(np.float32)
    va_g05 = ext_g05.transform(val.bank, val.arrays).astype(np.float32)
    struct_22_val = np.concatenate([va_g01, va_g03, va_g05], axis=-1)

    actual_cols = core_cols + symbolic_cols + g05_cols
    assert actual_cols == expected_22, f"Column order mismatch: {actual_cols} != {expected_22}"
    print(f"Feature extraction completed in {time.perf_counter() - t_feat:.2f}s (dim: {struct_22_train.shape[-1]}).")

    # 3. Step 3: Train MOD-M07a and generate 5-fold OOF
    print("\n--- Step 3: Generating 5-Fold OOF for MOD-M07a (Linear G05 View) ---")
    folds = kfold_indices(n_train, 5, FOLD_PARTITION_SEED)
    oof_v4_g05 = np.zeros((n_train, 4), dtype=np.float32)

    t_oof = time.perf_counter()
    for k, held in enumerate(folds):
        fit_rows = np.concatenate([folds[o] for o in range(5) if o != k])
        fit_hard = fit_rows[train_hard_mask[fit_rows]]

        scaler_k = StandardScaler()
        x_tr_hard_scaled = scaler_k.fit_transform(struct_22_train[fit_hard].reshape(-1, 22))
        y_tr_hard_flat = train.labels[fit_hard].reshape(-1)

        clf_k = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf_k.fit(x_tr_hard_scaled, y_tr_hard_flat)

        x_held_scaled = scaler_k.transform(struct_22_train[held].reshape(-1, 22))
        oof_v4_g05[held] = clf_k.decision_function(x_held_scaled).reshape(-1, 4)

    print(f"5-Fold OOF for MOD-M07a completed in {time.perf_counter() - t_oof:.2f}s.")

    # Fit full MOD-M07a on train_hard
    print("Fitting full MOD-M07a on complete train_hard (2,114 questions)...")
    scaler_full = StandardScaler()
    x_full_tr_hard = scaler_full.fit_transform(struct_22_train[train_hard_mask].reshape(-1, 22))
    y_full_tr_hard = train.labels[train_hard_mask].reshape(-1)

    clf_full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf_full.fit(x_full_tr_hard, y_full_tr_hard)

    x_val_scaled = scaler_full.transform(struct_22_val.reshape(-1, 22))
    val_g05_logits = clf_full.decision_function(x_val_scaled).reshape(-1, 4)

    # Standalone evaluation of MOD-M07a on validation set
    z_g05_masked = apply_validity_mask(val_g05_logits, val.bank, penalty=1e9)
    z_g05_masked = apply_conflict_penalty(z_g05_masked, val.bank, penalty=1e9)
    c_g05_arg = np.argmax(z_g05_masked, axis=1)
    acc_g05_arg = evaluate_choices(c_g05_arg, val.bank, gold_answers, is_hard)

    aggregator_g05 = SoftmaxClusterAggregator(temperature=0.5)
    c_g05_soft = aggregator_g05.aggregate(val.bank, z_g05_masked)
    acc_g05_soft = evaluate_choices(c_g05_soft, val.bank, gold_answers, is_hard)

    print(f"MOD-M07a Standalone Validation: argmax {acc_g05_arg['val_all']['correct']}/798 (hard {acc_g05_arg['val_hard']['correct']}/355), "
          f"softmax(T=0.5) {acc_g05_soft['val_all']['correct']}/798 (hard {acc_g05_soft['val_hard']['correct']}/355)")

    coef_dict = {col: float(clf_full.coef_[0, i]) for i, col in enumerate(actual_cols)}
    print("Top Positive Weights in MOD-M07a:")
    for col, w in sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col:30s}: {w:+.4f}")
    print("Top Negative Weights in MOD-M07a:")
    for col, w in sorted(coef_dict.items(), key=lambda x: x[1])[:5]:
        print(f"  {col:30s}: {w:+.4f}")

    # 4. Step 4: Stacking Weights Optimization on Train OOF
    print("\n--- Step 4: Optimizing 4-View Stacking Weights on Training OOF ---")

    # Load B17 / B20 OOF features
    # Arm A (v1) and Arm B (v2) from exp_b12
    oof_v1 = np.zeros((n_train, 4), dtype=np.float32)
    oof_v2 = np.zeros((n_train, 4), dtype=np.float32)
    for k, held in enumerate(folds):
        xa_held = train.emb[held]
        wa = dict(np.load(RUNS / f"exp_b12/seed_42/arm_embeddings_only_fold{k}.npz"))
        wb = dict(np.load(RUNS / f"exp_b12/seed_42/arm_embeddings_plus_g01_g03_fold{k}.npz"))
        oof_v1[held] = CrossCandidateAttentionNet(wa).forward_numpy(xa_held)
        m, std = wb["struct_mean"], wb["struct_std"]
        struct_16_train = struct_22_train[:, :, :16]
        struct_held_norm = (struct_16_train[held] - m) / np.maximum(std, 1e-6)
        xb_held = np.concatenate([train.emb[held], struct_held_norm], axis=-1)
        oof_v2[held] = CrossCandidateAttentionNet(wb).forward_numpy(xb_held)

    # Historical Tree R5 OOF
    struct_16_train = struct_22_train[:, :, :16]
    svd_tr_full = np.load(RUNS / "exp_b15/feat_g04b_svd.npz")["components"]
    proj_tr = (train.emb.reshape(-1, 1024) @ svd_tr_full.T).reshape(n_train, 4, 64)
    svd_128_tr = np.concatenate([proj_tr, proj_tr - proj_tr.mean(axis=1, keepdims=True)], axis=-1)
    mean_s16, std_s16 = np.mean(struct_16_train, axis=(0, 1)), np.std(struct_16_train, axis=(0, 1))
    x_tree_tr_std = np.concatenate([(struct_16_train - mean_s16) / np.maximum(std_s16, 1e-6), svd_128_tr], axis=-1)

    oof_tree_hist = np.zeros((n_train, 4), dtype=np.float32)
    for k, held in enumerate(folds):
        fit_rows = np.concatenate([folds[o] for o in range(5) if o != k])
        tree_k = HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS)
        tree_k.fit(x_tree_tr_std[fit_rows].reshape(-1, 144), train.labels[fit_rows].reshape(-1))
        oof_tree_hist[held] = tree_k.decision_function(x_tree_tr_std[held].reshape(-1, 144)).reshape(-1, 4)

    # Standardize OOF logits within problem
    s_v1_oof = standardize_within_problem(oof_v1)
    s_v2_oof = standardize_within_problem(oof_v2)
    s_v4_tree_oof = standardize_within_problem(oof_tree_hist)
    s_v_g05_oof = standardize_within_problem(oof_v4_g05)

    # Fit 4-view weights on historical OOF
    w_4v_hist_oof = fit_convex_combination_weights(
        [s_v1_oof, s_v2_oof, s_v4_tree_oof, s_v_g05_oof],
        train.labels,
        subset_mask=train_hard_mask,
    )
    print(f"4-View Stacking Weights (with Historical OOF):")
    print(f"  Arm A:      {w_4v_hist_oof[0]:.4f}")
    print(f"  Arm B:      {w_4v_hist_oof[1]:.4f}")
    print(f"  Tree R5:    {w_4v_hist_oof[2]:.4f}")
    print(f"  Linear G05: {w_4v_hist_oof[3]:.4f}")

    # Strict OOF optimization
    # Reconstruct strict OOF for A and B with 3-seed fold averaging
    oof_v1_strict = np.zeros((n_train, 4), dtype=np.float32)
    oof_v2_strict = np.zeros((n_train, 4), dtype=np.float32)
    oof_tree_strict = np.zeros((n_train, 4), dtype=np.float32)

    for k, held in enumerate(folds):
        fit_rows = np.concatenate([folds[o] for o in range(5) if o != k])
        # Arm A 3-seed fold average
        logits_a_list = []
        for s in SEEDS:
            wa = dict(np.load(RUNS / f"exp_b12/seed_{s}/arm_embeddings_only_fold{k}.npz"))
            logits_a_list.append(CrossCandidateAttentionNet(wa).forward_numpy(train.emb[held]))
        oof_v1_strict[held] = np.mean(logits_a_list, axis=0)

        # Arm B 3-seed fold average
        logits_b_list = []
        for s in SEEDS:
            wb = dict(np.load(RUNS / f"exp_b12/seed_{s}/arm_embeddings_plus_g01_g03_fold{k}.npz"))
            m, std = wb["struct_mean"], wb["struct_std"]
            struct_held_norm = (struct_16_train[held] - m) / np.maximum(std, 1e-6)
            xb_held = np.concatenate([train.emb[held], struct_held_norm], axis=-1)
            logits_b_list.append(CrossCandidateAttentionNet(wb).forward_numpy(xb_held))
        oof_v2_strict[held] = np.mean(logits_b_list, axis=0)

        # Tree fold-local SVD + raw struct
        svd_k = TruncatedSVD(n_components=64, random_state=42)
        fit_flat_emb = train.emb[fit_rows].reshape(-1, 1024)
        svd_k.fit(fit_flat_emb)
        fit_proj = (fit_flat_emb @ svd_k.components_.T).reshape(len(fit_rows), 4, 64)
        fit_svd_128 = np.concatenate([fit_proj, fit_proj - fit_proj.mean(axis=1, keepdims=True)], axis=-1)
        held_flat_emb = train.emb[held].reshape(-1, 1024)
        held_proj = (held_flat_emb @ svd_k.components_.T).reshape(len(held), 4, 64)
        held_svd_128 = np.concatenate([held_proj, held_proj - held_proj.mean(axis=1, keepdims=True)], axis=-1)
        x_tree_fit = np.concatenate([struct_16_train[fit_rows], fit_svd_128], axis=-1)
        x_tree_held = np.concatenate([struct_16_train[held], held_svd_128], axis=-1)
        tree_fold = HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS)
        tree_fold.fit(x_tree_fit.reshape(-1, 144), train.labels[fit_rows].reshape(-1))
        oof_tree_strict[held] = tree_fold.decision_function(x_tree_held.reshape(-1, 144)).reshape(-1, 4)

    s_v1_strict_oof = standardize_within_problem(oof_v1_strict)
    s_v2_strict_oof = standardize_within_problem(oof_v2_strict)
    s_tree_strict_oof = standardize_within_problem(oof_tree_strict)

    w_4v_strict_oof = fit_convex_combination_weights(
        [s_v1_strict_oof, s_v2_strict_oof, s_tree_strict_oof, s_v_g05_oof],
        train.labels,
        subset_mask=train_hard_mask,
    )
    print(f"\n4-View Stacking Weights (with Strict OOF):")
    print(f"  Arm A:      {w_4v_strict_oof[0]:.4f}")
    print(f"  Arm B:      {w_4v_strict_oof[1]:.4f}")
    print(f"  Tree R5:    {w_4v_strict_oof[2]:.4f}")
    print(f"  Linear G05: {w_4v_strict_oof[3]:.4f}")

    # 5. Step 5: Validation Evaluation with Paired Seeds
    print("\n--- Step 5: Validation Evaluation (Paired Seeds & Final Ensemble) ---")
    logits_a_dict = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b_dict = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    tree_val_raw = np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")["seed_42"]

    s_tree_val = standardize_within_problem(tree_val_raw)
    s_g05_val = standardize_within_problem(val_g05_logits)

    # 3-seed paired evaluation
    paired_results = {}
    w3_b17 = [0.6519943983184848, 0.2773940600630942, 0.07061154161842091]

    for s in SEEDS:
        v1_s = logits_a_dict[f"seed_{s}_full"]
        v2_s = logits_b_dict[f"seed_{s}_full"]
        s1_s = standardize_within_problem(v1_s)
        s2_s = standardize_within_problem(v2_s)

        # 3-View baseline (without G05)
        stacked_3v = w3_b17[0] * s1_s + w3_b17[1] * s2_s + w3_b17[2] * s_tree_val
        z_3v = apply_validity_mask(stacked_3v, val.bank, penalty=1e9)
        z_3v = apply_conflict_penalty(z_3v, val.bank, penalty=1e9)
        c_3v = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_3v)
        acc_3v = evaluate_choices(c_3v, val.bank, gold_answers, is_hard)

        # 4-View candidate (with G05)
        stacked_4v = (
            w_4v_hist_oof[0] * s1_s
            + w_4v_hist_oof[1] * s2_s
            + w_4v_hist_oof[2] * s_tree_val
            + w_4v_hist_oof[3] * s_g05_val
        )
        z_4v = apply_validity_mask(stacked_4v, val.bank, penalty=1e9)
        z_4v = apply_conflict_penalty(z_4v, val.bank, penalty=1e9)
        c_4v = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_4v)
        acc_4v = evaluate_choices(c_4v, val.bank, gold_answers, is_hard)

        delta_hard = acc_4v["val_hard"]["correct"] - acc_3v["val_hard"]["correct"]
        delta_all = acc_4v["val_all"]["correct"] - acc_3v["val_all"]["correct"]

        paired_results[f"seed_{s}"] = {
            "3view_baseline": {"val_all": acc_3v["val_all"]["correct"], "val_hard": acc_3v["val_hard"]["correct"]},
            "4view_g05": {"val_all": acc_4v["val_all"]["correct"], "val_hard": acc_4v["val_hard"]["correct"]},
            "delta_hard": delta_hard,
            "delta_all": delta_all,
        }
        print(f"  Seed {s} Pair: 3V={acc_3v['val_all']['correct']} (hard {acc_3v['val_hard']['correct']}) -> "
              f"4V={acc_4v['val_all']['correct']} (hard {acc_4v['val_hard']['correct']}) | "
              f"Delta hard: {delta_hard:+d}, all: {delta_all:+d}")

    # Final Ensemble Evaluation (3-seed averaged views)
    s1_ens = standardize_within_problem(logits_a_dict["seed_ensemble"])
    s2_ens = standardize_within_problem(logits_b_dict["seed_ensemble"])

    # 3-View confirmed baseline
    stacked_3v_ens = w3_b17[0] * s1_ens + w3_b17[1] * s2_ens + w3_b17[2] * s_tree_val
    z_3v_ens = apply_validity_mask(stacked_3v_ens, val.bank, penalty=1e9)
    z_3v_ens = apply_conflict_penalty(z_3v_ens, val.bank, penalty=1e9)
    c_3v_ens = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_3v_ens)
    acc_3v_ens = evaluate_choices(c_3v_ens, val.bank, gold_answers, is_hard)

    # 4-View candidate (Historical OOF weights)
    stacked_4v_ens = (
        w_4v_hist_oof[0] * s1_ens
        + w_4v_hist_oof[1] * s2_ens
        + w_4v_hist_oof[2] * s_tree_val
        + w_4v_hist_oof[3] * s_g05_val
    )
    z_4v_ens = apply_validity_mask(stacked_4v_ens, val.bank, penalty=1e9)
    z_4v_ens = apply_conflict_penalty(z_4v_ens, val.bank, penalty=1e9)
    c_4v_ens = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_4v_ens)
    acc_4v_ens = evaluate_choices(c_4v_ens, val.bank, gold_answers, is_hard)

    # 4-View candidate (Strict OOF weights)
    stacked_4v_strict = (
        w_4v_strict_oof[0] * s1_ens
        + w_4v_strict_oof[1] * s2_ens
        + w_4v_strict_oof[2] * s_tree_val
        + w_4v_strict_oof[3] * s_g05_val
    )
    z_4v_strict = apply_validity_mask(stacked_4v_strict, val.bank, penalty=1e9)
    z_4v_strict = apply_conflict_penalty(z_4v_strict, val.bank, penalty=1e9)
    c_4v_strict = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_4v_strict)
    acc_4v_strict = evaluate_choices(c_4v_strict, val.bank, gold_answers, is_hard)

    print("\n" + "=" * 78)
    print("Final Ensemble Comparison:")
    print(f"  3-View Confirmed (B17/B20):  val_all = {acc_3v_ens['val_all']['correct']} ({acc_3v_ens['val_all']['accuracy']:.2f}%), "
          f"val_hard = {acc_3v_ens['val_hard']['correct']} ({acc_3v_ens['val_hard']['accuracy']:.2f}%)")
    print(f"  4-View + G05 (Hist Weights): val_all = {acc_4v_ens['val_all']['correct']} ({acc_4v_ens['val_all']['accuracy']:.2f}%), "
          f"val_hard = {acc_4v_ens['val_hard']['correct']} ({acc_4v_ens['val_hard']['accuracy']:.2f}%)")
    print(f"  4-View + G05 (Strict Weights): val_all = {acc_4v_strict['val_all']['correct']} ({acc_4v_strict['val_all']['accuracy']:.2f}%), "
          f"val_hard = {acc_4v_strict['val_hard']['correct']} ({acc_4v_strict['val_hard']['accuracy']:.2f}%)")
    print("=" * 78)

    # 6. Step 6: Pre-registered Retention Gate Check
    # Rule 1: Paired seed mean delta(val_hard) >= +3.55 questions (+1.0%)
    # Rule 2: At least 2 of 3 seeds delta(val_hard) > 0
    # Rule 3: Paired seed mean delta(val_all) >= 0.0
    # Rule 4: Ensemble delta(val_hard) >= +4.0 questions vs B17 benchmark (300)
    # Rule 5: Ensemble val_all >= B17 benchmark (743)
    paired_hard_deltas = [r["delta_hard"] for r in paired_results.values()]
    paired_all_deltas = [r["delta_all"] for r in paired_results.values()]

    mean_paired_hard_delta = float(np.mean(paired_hard_deltas))
    positive_hard_count = sum(d > 0 for d in paired_hard_deltas)
    mean_paired_all_delta = float(np.mean(paired_all_deltas))

    ens_hard_gain = acc_4v_ens["val_hard"]["correct"] - 300
    ens_all_gain = acc_4v_ens["val_all"]["correct"] - 743

    gate_r1 = bool(mean_paired_hard_delta >= 3.55)
    gate_r2 = bool(positive_hard_count >= 2)
    gate_r3 = bool(mean_paired_all_delta >= 0.0)
    gate_r4 = bool(ens_hard_gain >= 4)
    gate_r5 = bool(ens_all_gain >= 0)

    retained = gate_r1 and gate_r2 and gate_r3 and gate_r4 and gate_r5
    verdict = "RETAINED" if retained else "REJECTED"

    print("\nEXP-B21 Retention Gate Evaluation:")
    print(f"  [Gate 1] Paired hard delta mean >= +3.55: {mean_paired_hard_delta:+.2f} ({'PASS' if gate_r1 else 'FAIL'})")
    print(f"  [Gate 2] Paired hard delta positive count >= 2: {positive_hard_count}/3 ({'PASS' if gate_r2 else 'FAIL'})")
    print(f"  [Gate 3] Paired all delta mean >= 0.0: {mean_paired_all_delta:+.2f} ({'PASS' if gate_r3 else 'FAIL'})")
    print(f"  [Gate 4] Final ensemble hard gain vs 300 >= +4: {ens_hard_gain:+d} ({'PASS' if gate_r4 else 'FAIL'})")
    print(f"  [Gate 5] Final ensemble all gain vs 743 >= 0: {ens_all_gain:+d} ({'PASS' if gate_r5 else 'FAIL'})")
    print(f"  >>> FINAL VERDICT: {verdict} <<<")

    # 7. Step 7: Official Prediction & Scoring Output
    pred_path_4v = EXP_B21_DIR / "EXP-B21_4view_predictions.json"
    score_path_4v = EXP_B21_DIR / "EXP-B21_4view_score.json"
    write_official_predictions(pred_path_4v, val.bank, c_4v_ens)
    sc_4v = official_score(pred_path_4v, score_path_4v)

    # 8. Asset footprint of MOD-M07a
    # Coefficients: 22 float64 + 1 bias float64 = 23 * 8 = 184 bytes
    # Scaler: 22 mean float64 + 22 scale float64 = 352 bytes
    linear_asset_bytes = 184 + 352

    # 9. Write Comprehensive Audit Log
    audit_data = {
        "experiment": "EXP-B21",
        "description": "G05 Fine-Grained Symbolic View Integration & Stacking Ablation",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_record(torch.device("cpu")),
        "model_id": "MOD-M07a",
        "columns": actual_cols,
        "column_count": len(actual_cols),
        "mod_m07a_coefficients": coef_dict,
        "mod_m07a_intercept": float(clf_full.intercept_[0]),
        "mod_m07a_standalone": {
            "argmax": acc_g05_arg,
            "softmax_t05": acc_g05_soft,
        },
        "training_oof_optimization": {
            "historical_oof_weights": {
                "arm_a": float(w_4v_hist_oof[0]),
                "arm_b": float(w_4v_hist_oof[1]),
                "tree_r5": float(w_4v_hist_oof[2]),
                "linear_g05": float(w_4v_hist_oof[3]),
            },
            "strict_oof_weights": {
                "arm_a": float(w_4v_strict_oof[0]),
                "arm_b": float(w_4v_strict_oof[1]),
                "tree_r5": float(w_4v_strict_oof[2]),
                "linear_g05": float(w_4v_strict_oof[3]),
            },
        },
        "paired_seed_evaluation": paired_results,
        "final_ensemble_evaluation": {
            "3view_b17_baseline": {
                "val_all": acc_3v_ens["val_all"]["correct"],
                "val_hard": acc_3v_ens["val_hard"]["correct"],
                "accuracy_all": acc_3v_ens["val_all"]["accuracy"],
                "accuracy_hard": acc_3v_ens["val_hard"]["accuracy"],
            },
            "4view_g05_hist_weights": {
                "val_all": acc_4v_ens["val_all"]["correct"],
                "val_hard": acc_4v_ens["val_hard"]["correct"],
                "accuracy_all": acc_4v_ens["val_all"]["accuracy"],
                "accuracy_hard": acc_4v_ens["val_hard"]["accuracy"],
            },
            "4view_g05_strict_weights": {
                "val_all": acc_4v_strict["val_all"]["correct"],
                "val_hard": acc_4v_strict["val_hard"]["correct"],
                "accuracy_all": acc_4v_strict["val_all"]["accuracy"],
                "accuracy_hard": acc_4v_strict["val_hard"]["accuracy"],
            },
        },
        "retention_gate": {
            "rule1_paired_hard_delta_ge_3_55": {"value": mean_paired_hard_delta, "passed": gate_r1},
            "rule2_positive_hard_count_ge_2": {"value": positive_hard_count, "passed": gate_r2},
            "rule3_paired_all_delta_ge_0": {"value": mean_paired_all_delta, "passed": gate_r3},
            "rule4_ensemble_hard_gain_ge_4": {"value": ens_hard_gain, "passed": gate_r4},
            "rule5_ensemble_all_gain_ge_0": {"value": ens_all_gain, "passed": gate_r5},
            "verdict": verdict,
        },
        "linear_view_asset_bytes": linear_asset_bytes,
        "official_score": sc_4v,
        "execution_seconds": time.perf_counter() - t0,
    }

    audit_path = AUDIT_DIR / "exp_b21_g05_view_audit.json"
    audit_path.write_text(json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")
    print(f"\nAudit log saved to: {audit_path}")

    return audit_data


if __name__ == "__main__":
    run_exp_b21()
