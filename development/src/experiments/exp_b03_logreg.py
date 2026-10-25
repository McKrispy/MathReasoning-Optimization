"""EXP-B03: Standardized L2 Logistic Regression & Rare Feature Contingency Tables.

Evaluates feature directional impact (pushing candidates towards 'correct' or 'wrong').
Performs:
1. 2x2 Contingency tables for rare binary columns:
   Reads contingency_table_columns from dev/runs/audit_logs/private_configuration_omitted.json.
   Compares train_hard and val_hard conditional accuracies to check for direction consistency.
2. Fit 1: Standardized L2 Logistic Regression on Consensus Core columns.
   Loaded dynamically from contract: statistically_eligible_by_family['FEAT-G01'] minus gsm8k_morphology_columns_not_core.
   Reports weights and val_hard accuracy vs Majority Voting (69.0141%).
3. Fit 2: Standardized L2 Logistic Regression on Consensus Core + other eligible family columns.
   Loaded dynamically from contract: Fit 1 cols + other families in statistically_eligible_columns
   excluding gsm8k_morphology_columns_not_core and contingency_table_columns.
   Reports weights and val_hard accuracy.
"""
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import (
    load_dataset,
    load_gold_references,
    get_hard_subset_mask,
    build_label_matrix,
)
from features.consensus import ConsensusFeatureExtractor
from features.surface import SurfaceFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor
from features.geometry import GeometryFeatureExtractor


def run_exp_b03() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B03: Standardized L2 Logistic Regression & Contingency Tables")
    print("=" * 70)

    # 1. Load column contract (Authority for downstream column lists)
    contract_path = DEV_SRC_DIR.parent / 'runs' / 'audit_logs' / 'private_configuration_omitted.json'
    if not contract_path.exists():
        raise FileNotFoundError(f"Column contract not found at: {contract_path}")
    
    with open(contract_path, 'r', encoding='utf-8') as f:
        contract = json.load(f)
    print(f"Loaded column contract from: {contract_path}")
    print(f"Total statistically eligible columns in contract: {contract['statistically_eligible_count']}")

    # 2. Load data
    bank_train, arrays_train = load_dataset('train')
    gold_train = load_gold_references('train')
    train_hard_mask = get_hard_subset_mask(bank_train)
    bank_train_hard = [b for i, b in enumerate(bank_train) if train_hard_mask[i]]
    arrays_train_hard = {'embeddings': arrays_train['embeddings'][train_hard_mask]}
    labels_train_hard = build_label_matrix(bank_train_hard, [g for i, g in enumerate(gold_train) if train_hard_mask[i]])

    bank_val, arrays_val = load_dataset('validation')
    gold_val = load_gold_references('validation')
    val_hard_mask = get_hard_subset_mask(bank_val)
    bank_val_hard = [b for i, b in enumerate(bank_val) if val_hard_mask[i]]
    arrays_val_hard = {'embeddings': arrays_val['embeddings'][val_hard_mask]}
    labels_val_hard = build_label_matrix(bank_val_hard, [g for i, g in enumerate(gold_val) if val_hard_mask[i]])
    labels_val_all = build_label_matrix(bank_val, gold_val)

    n_train = len(bank_train_hard)
    n_val_hard = len(bank_val_hard)
    n_val_all = len(bank_val)
    print(f"Loaded train_hard: {n_train} questions ({n_train*4} candidates)")
    print(f"Loaded val_hard:   {n_val_hard} questions ({n_val_hard*4} candidates)")
    print(f"Loaded val_all:    {n_val_all} questions ({n_val_all*4} candidates)")

    # 3. Extract features
    ext_g01 = ConsensusFeatureExtractor()
    ext_g02 = SurfaceFeatureExtractor()
    ext_g03 = SymbolicVerificationFeatureExtractor()
    ext_g04 = GeometryFeatureExtractor()

    print("\nExtracting features for train_hard...")
    x_tr_g01 = ext_g01.transform(bank_train_hard, arrays_train_hard)
    x_tr_g02 = ext_g02.transform(bank_train_hard, arrays_train_hard)
    x_tr_g03 = ext_g03.transform(bank_train_hard, arrays_train_hard)
    x_tr_g04 = ext_g04.transform(bank_train_hard, arrays_train_hard)
    x_tr_all = np.concatenate([x_tr_g01, x_tr_g02, x_tr_g03, x_tr_g04], axis=-1)

    print("Extracting features for val_hard...")
    x_vh_g01 = ext_g01.transform(bank_val_hard, arrays_val_hard)
    x_vh_g02 = ext_g02.transform(bank_val_hard, arrays_val_hard)
    x_vh_g03 = ext_g03.transform(bank_val_hard, arrays_val_hard)
    x_vh_g04 = ext_g04.transform(bank_val_hard, arrays_val_hard)
    x_vh_all = np.concatenate([x_vh_g01, x_vh_g02, x_vh_g03, x_vh_g04], axis=-1)

    print("Extracting features for val_all...")
    x_va_g01 = ext_g01.transform(bank_val, arrays_val)
    x_va_g02 = ext_g02.transform(bank_val, arrays_val)
    x_va_g03 = ext_g03.transform(bank_val, arrays_val)
    x_va_g04 = ext_g04.transform(bank_val, arrays_val)
    x_va_all = np.concatenate([x_va_g01, x_va_g02, x_va_g03, x_va_g04], axis=-1)

    all_feature_names = (
        ext_g01.FEATURE_NAMES +
        ext_g02.FEATURE_NAMES +
        ext_g03.FEATURE_NAMES +
        ext_g04.FEATURE_NAMES
    )
    feat2idx = {name: i for i, name in enumerate(all_feature_names)}

    # 4. Part A: Contingency Tables for Rare Binary Columns (from contract)
    contingency_cols = contract["contingency_table_columns"]
    contingency_results = []
    cols_to_delete = []

    print("\n" + "=" * 70)
    print("PART A: CONTINGENCY TABLES FOR RARE 0/1 COLUMNS")
    print("=" * 70)

    y_tr_flat = labels_train_hard.reshape(-1)
    y_vh_flat = labels_val_hard.reshape(-1)

    for col in contingency_cols:
        c_idx = feat2idx[col]
        x_tr_col = x_tr_all.reshape(-1, 56)[:, c_idx]
        x_vh_col = x_vh_all.reshape(-1, 56)[:, c_idx]

        # Train hard stats
        tr_trig = (x_tr_col > 0.5)
        tr_n_trig = int(np.sum(tr_trig))
        tr_acc_trig = float(np.mean(y_tr_flat[tr_trig])) if tr_n_trig > 0 else 0.0
        tr_n_not = int(np.sum(~tr_trig))
        tr_acc_not = float(np.mean(y_tr_flat[~tr_trig])) if tr_n_not > 0 else 0.0
        tr_diff = tr_acc_trig - tr_acc_not

        # Val hard stats
        vh_trig = (x_vh_col > 0.5)
        vh_n_trig = int(np.sum(vh_trig))
        vh_acc_trig = float(np.mean(y_vh_flat[vh_trig])) if vh_n_trig > 0 else 0.0
        vh_n_not = int(np.sum(~vh_trig))
        vh_acc_not = float(np.mean(y_vh_flat[~vh_trig])) if vh_n_not > 0 else 0.0
        vh_diff = vh_acc_trig - vh_acc_not

        # Sign consistency check
        opposite_direction = (tr_diff * vh_diff < 0)
        action = "DELETE (Inconsistent Direction)" if opposite_direction else "CONSISTENT"
        if opposite_direction:
            cols_to_delete.append(col)

        res_entry = {
            "feature": col,
            "train_hard": {
                "triggered_count": tr_n_trig,
                "triggered_acc": tr_acc_trig,
                "not_triggered_count": tr_n_not,
                "not_triggered_acc": tr_acc_not,
                "diff": tr_diff,
            },
            "val_hard": {
                "triggered_count": vh_n_trig,
                "triggered_acc": vh_acc_trig,
                "not_triggered_count": vh_n_not,
                "not_triggered_acc": vh_acc_not,
                "diff": vh_diff,
            },
            "opposite_direction": opposite_direction,
            "action": action,
        }
        contingency_results.append(res_entry)

        print(f"\nFeature: `{col}`")
        print(f"  train_hard: trig={tr_n_trig}/{len(y_tr_flat)} ({tr_n_trig/len(y_tr_flat)*100:.2f}%), Acc(trig)={tr_acc_trig*100:.2f}% vs Acc(not)={tr_acc_not*100:.2f}% -> Diff={tr_diff*100:+.2f}%")
        print(f"  val_hard:   trig={vh_n_trig}/{len(y_vh_flat)} ({vh_n_trig/len(y_vh_flat)*100:.2f}%), Acc(trig)={vh_acc_trig*100:.2f}% vs Acc(not)={vh_acc_not*100:.2f}% -> Diff={vh_diff*100:+.2f}%")
        print(f"  --> Evaluation: {action}")

    # 5. Determine Feature Sets strictly from Contract
    # Fit 1: Consensus Core
    morphology_set = set(contract["gsm8k_morphology_columns_not_core"])
    contingency_set = set(contract["contingency_table_columns"])

    fit1_cols = [
        c for c in contract["statistically_eligible_by_family"]["FEAT-G01"]
        if c not in morphology_set
    ]

    # Fit 2: Consensus Core + other families
    other_eligible_cols = [
        c for c in contract["statistically_eligible_columns"]
        if c not in contract["statistically_eligible_by_family"]["FEAT-G01"]
        and c not in morphology_set
        and c not in contingency_set
    ]
    fit2_cols = fit1_cols + other_eligible_cols

    print(f"\n[Contract-Resolved Feature Lists]")
    print(f"  Fit 1 Columns ({len(fit1_cols)} dims): {fit1_cols}")
    print(f"  Fit 2 Columns ({len(fit2_cols)} dims): {fit2_cols}")

    # 6. Fit 1: Consensus Core Logistic Regression
    print("\n" + "=" * 70)
    print(f"PART B: FIT 1 - CONSENSUS CORE LOGISTIC REGRESSION ({len(fit1_cols)} features)")
    print("=" * 70)

    fit1_indices = [feat2idx[c] for c in fit1_cols]
    scaler_fit1 = StandardScaler()
    x_tr_fit1 = scaler_fit1.fit_transform(x_tr_all.reshape(-1, 56)[:, fit1_indices])
    x_vh_fit1 = scaler_fit1.transform(x_vh_all.reshape(-1, 56)[:, fit1_indices])
    x_va_fit1 = scaler_fit1.transform(x_va_all.reshape(-1, 56)[:, fit1_indices])

    clf_fit1 = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf_fit1.fit(x_tr_fit1, y_tr_flat)

    fit1_weights = []
    for c_name, w in zip(fit1_cols, clf_fit1.coef_[0]):
        fit1_weights.append({"feature": c_name, "weight": float(w)})
        print(f"  {c_name:<25}: Weight = {w:+.4f} ({'Positive' if w > 0 else 'Negative'})")

    # Evaluate question-level accuracy on val_hard
    scores_vh_fit1 = clf_fit1.decision_function(x_vh_fit1).reshape(n_val_hard, 4)
    pred_vh_fit1 = np.argmax(scores_vh_fit1, axis=1)
    acc_vh_fit1 = float(np.mean([labels_val_hard[i, pred_vh_fit1[i]] for i in range(n_val_hard)]))

    # Evaluate question-level accuracy on val_all
    scores_va_fit1 = clf_fit1.decision_function(x_va_fit1).reshape(n_val_all, 4)
    pred_va_fit1 = np.argmax(scores_va_fit1, axis=1)
    acc_va_fit1 = float(np.mean([labels_val_all[i, pred_va_fit1[i]] for i in range(n_val_all)]))

    # Evaluate on train_hard
    scores_tr_fit1 = clf_fit1.decision_function(x_tr_fit1).reshape(n_train, 4)
    pred_tr_fit1 = np.argmax(scores_tr_fit1, axis=1)
    acc_tr_fit1 = float(np.mean([labels_train_hard[i, pred_tr_fit1[i]] for i in range(n_train)]))

    print(f"\nFit 1 Accuracy Summary:")
    print(f"  val_hard:   {acc_vh_fit1*100:.4f}% ({int(acc_vh_fit1*n_val_hard)} / {n_val_hard}) [Majority Voting: 69.0141%]")
    print(f"  val_all:    {acc_va_fit1*100:.4f}% ({int(acc_va_fit1*n_val_all)} / {n_val_all}) [Majority Voting: 86.2155%]")
    print(f"  train_hard: {acc_tr_fit1*100:.4f}% ({int(acc_tr_fit1*n_train)} / {n_train}) [Majority Voting: 66.8398%]")

    # 7. Fit 2: Consensus Core + Retained Features
    print("\n" + "=" * 70)
    print(f"PART C: FIT 2 - FULL RETAINED FEATURES LOGISTIC REGRESSION ({len(fit2_cols)} features)")
    print("=" * 70)

    fit2_indices = [feat2idx[c] for c in fit2_cols]
    scaler_fit2 = StandardScaler()
    x_tr_fit2 = scaler_fit2.fit_transform(x_tr_all.reshape(-1, 56)[:, fit2_indices])
    x_vh_fit2 = scaler_fit2.transform(x_vh_all.reshape(-1, 56)[:, fit2_indices])
    x_va_fit2 = scaler_fit2.transform(x_va_all.reshape(-1, 56)[:, fit2_indices])

    clf_fit2 = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    clf_fit2.fit(x_tr_fit2, y_tr_flat)

    fit2_weights = []
    for c_name, w in zip(fit2_cols, clf_fit2.coef_[0]):
        fit2_weights.append({"feature": c_name, "weight": float(w)})
    fit2_weights.sort(key=lambda x: x["weight"], reverse=True)

    print("\nRanked Feature Weights (Positive -> Negative pushers):")
    for item in fit2_weights:
        direction = "++ POSITIVE PUSHER" if item["weight"] > 0.05 else ("-- NEGATIVE PUSHER" if item["weight"] < -0.05 else "   NEUTRAL")
        print(f"  {item['feature']:<30}: Weight = {item['weight']:+.4f}  [{direction}]")

    # Evaluate question-level accuracy on val_hard
    scores_vh_fit2 = clf_fit2.decision_function(x_vh_fit2).reshape(n_val_hard, 4)
    pred_vh_fit2 = np.argmax(scores_vh_fit2, axis=1)
    acc_vh_fit2 = float(np.mean([labels_val_hard[i, pred_vh_fit2[i]] for i in range(n_val_hard)]))

    # Evaluate question-level accuracy on val_all
    scores_va_fit2 = clf_fit2.decision_function(x_va_fit2).reshape(n_val_all, 4)
    pred_va_fit2 = np.argmax(scores_va_fit2, axis=1)
    acc_va_fit2 = float(np.mean([labels_val_all[i, pred_va_fit2[i]] for i in range(n_val_all)]))

    # Evaluate on train_hard
    scores_tr_fit2 = clf_fit2.decision_function(x_tr_fit2).reshape(n_train, 4)
    pred_tr_fit2 = np.argmax(scores_tr_fit2, axis=1)
    acc_tr_fit2 = float(np.mean([labels_train_hard[i, pred_tr_fit2[i]] for i in range(n_train)]))

    print(f"\nFit 2 Accuracy Summary:")
    print(f"  val_hard:   {acc_vh_fit2*100:.4f}% ({int(acc_vh_fit2*n_val_hard)} / {n_val_hard}) [Fit 1: {acc_vh_fit1*100:.4f}%, MV: 69.0141%]")
    print(f"  val_all:    {acc_va_fit2*100:.4f}% ({int(acc_va_fit2*n_val_all)} / {n_val_all}) [Fit 1: {acc_va_fit1*100:.4f}%, MV: 86.2155%]")
    print(f"  train_hard: {acc_tr_fit2*100:.4f}% ({int(acc_tr_fit2*n_train)} / {n_train}) [Fit 1: {acc_tr_fit1*100:.4f}%, MV: 66.8398%]")

    results = {
        "dataset": "train_hard / val_hard",
        "contingency_tables": contingency_results,
        "cols_to_delete": cols_to_delete,
        "fit1_consensus_core": {
            "num_features": len(fit1_cols),
            "features": fit1_cols,
            "weights": fit1_weights,
            "acc_val_hard": acc_vh_fit1,
            "acc_val_all": acc_va_fit1,
            "acc_train_hard": acc_tr_fit1,
        },
        "fit2_full_retained": {
            "num_features": len(fit2_cols),
            "features": fit2_cols,
            "weights": fit2_weights,
            "acc_val_hard": acc_vh_fit2,
            "acc_val_all": acc_va_fit2,
            "acc_train_hard": acc_tr_fit2,
        },
    }

    # Save to disk
    runs_dir = DEV_SRC_DIR.parent / 'runs' / 'audit_logs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_file = runs_dir / 'exp_b03_logreg_audit.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull audit saved to: {out_file}")

    return results


if __name__ == '__main__':
    run_exp_b03()
