"""EXP-B04: Incremental Feature Family Ablation.

Evaluates incremental contribution of each feature family over Consensus Core baseline.
Reads feature specifications strictly from dev/runs/audit_logs/private_configuration_omitted.json.

Uses identical standardized L2 Logistic Regression ranking model (C=1.0, lbfgs, max_iter=1000).
Fits on train_hard (2,114 questions), evaluates question-level candidate selection on val_hard (355 questions)
and val_all (798 questions).

Four Ablation Runs over Baseline (Consensus Core, 4 features):
  Run 1: Baseline + FEAT-G02 (Surface Length, Formatting & Wording Heuristics)
  Run 2: Baseline + FEAT-G03 (Reasoning & Symbolic Verification)
  Run 3: Baseline + FEAT-G04 (Supplied Representations & Cross-Candidate Geometry)
  Run 4: Baseline + GSM8K Answer Morphology (5 columns)

Gate Criterion (Pre-committed in private_document_omitted):
  A family enters EXP-B09 and EXP-B10 if and only if:
    Delta(val_hard) >= +1.0 percentage point (~4 questions out of 355)
    AND val_all does not drop significantly.
  GSM8K Answer Morphology, even if Delta >= +1.0%, is retained only as a low-weight contrast.
"""
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
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


def run_exp_b04_ablation() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B04: Incremental Feature Family Ablation")
    print("=" * 70)

    # 1. Load column contract (Authority for downstream column lists)
    contract_path = DEV_SRC_DIR.parent / 'runs' / 'audit_logs' / 'private_configuration_omitted.json'
    if not contract_path.exists():
        raise FileNotFoundError(f"Column contract JSON not found at: {contract_path}")
    
    with open(contract_path, 'r', encoding='utf-8') as f:
        contract = json.load(f)
    print(f"Loaded column contract from: {contract_path}")

    # 2. Load dataset
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
    y_tr_flat = labels_train_hard.reshape(-1)

    # 4. Define Feature Sets strictly from Contract
    morphology_set = set(contract["gsm8k_morphology_columns_not_core"])
    contingency_set = set(contract["contingency_table_columns"])

    # Baseline: Consensus Core (Fit 1 from EXP-B03)
    base_cols = [
        c for c in contract["statistically_eligible_by_family"]["FEAT-G01"]
        if c not in morphology_set
    ]

    # Family 1: FEAT-G02 Surface
    feat_g02_cols = contract["statistically_eligible_by_family"]["FEAT-G02"]

    # Family 2: FEAT-G03 Symbolic & Reasoning (eligible)
    feat_g03_cols = contract["statistically_eligible_by_family"]["FEAT-G03"]
    # Also test regression-safe version without contingency table column
    feat_g03_reg_cols = [c for c in feat_g03_cols if c not in contingency_set]

    # Family 3: FEAT-G04 Geometry
    feat_g04_cols = contract["statistically_eligible_by_family"]["FEAT-G04"]

    # Family 4: GSM8K Morphology (5 dimensions)
    morphology_cols = [c for c in contract["gsm8k_morphology_columns_not_core"]]

    print(f"\n[Feature Configuration from Contract]")
    print(f"  Baseline (Consensus Core): {len(base_cols)} cols -> {base_cols}")
    print(f"  FEAT-G02 (Surface):        {len(feat_g02_cols)} cols -> {feat_g02_cols}")
    print(f"  FEAT-G03 (Symbolic):       {len(feat_g03_cols)} cols -> {feat_g03_cols}")
    print(f"  FEAT-G04 (Geometry):       {len(feat_g04_cols)} cols -> {feat_g04_cols}")
    print(f"  GSM8K Morphology:          {len(morphology_cols)} cols -> {morphology_cols}")

    # Helper evaluation function
    def evaluate_model(feature_cols: List[str], run_name: str) -> Dict[str, Any]:
        indices = [feat2idx[c] for c in feature_cols]
        scaler = StandardScaler()
        x_tr_sub = scaler.fit_transform(x_tr_all.reshape(-1, 56)[:, indices])
        x_vh_sub = scaler.transform(x_vh_all.reshape(-1, 56)[:, indices])
        x_va_sub = scaler.transform(x_va_all.reshape(-1, 56)[:, indices])

        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(x_tr_sub, y_tr_flat)

        # val_hard question-level accuracy
        scores_vh = clf.decision_function(x_vh_sub).reshape(n_val_hard, 4)
        pred_vh = np.argmax(scores_vh, axis=1)
        acc_vh = float(np.mean([labels_val_hard[i, pred_vh[i]] for i in range(n_val_hard)]))
        correct_vh = int(np.sum([labels_val_hard[i, pred_vh[i]] for i in range(n_val_hard)]))

        # val_all question-level accuracy
        scores_va = clf.decision_function(x_va_sub).reshape(n_val_all, 4)
        pred_va = np.argmax(scores_va, axis=1)
        acc_va = float(np.mean([labels_val_all[i, pred_va[i]] for i in range(n_val_all)]))
        correct_va = int(np.sum([labels_val_all[i, pred_va[i]] for i in range(n_val_all)]))

        # train_hard question-level accuracy
        scores_tr = clf.decision_function(x_tr_sub).reshape(n_train, 4)
        pred_tr = np.argmax(scores_tr, axis=1)
        acc_tr = float(np.mean([labels_train_hard[i, pred_tr[i]] for i in range(n_train)]))
        correct_tr = int(np.sum([labels_train_hard[i, pred_tr[i]] for i in range(n_train)]))

        weights = {c: float(w) for c, w in zip(feature_cols, clf.coef_[0])}

        return {
            "run_name": run_name,
            "num_features": len(feature_cols),
            "features": feature_cols,
            "weights": weights,
            "val_hard": {
                "acc": acc_vh,
                "correct": correct_vh,
                "total": n_val_hard,
            },
            "val_all": {
                "acc": acc_va,
                "correct": correct_va,
                "total": n_val_all,
            },
            "train_hard": {
                "acc": acc_tr,
                "correct": correct_tr,
                "total": n_train,
            },
        }

    # 5. Run Baseline
    print("\n" + "=" * 70)
    print("RUN 0: BASELINE - CONSENSUS CORE ONLY (4 features)")
    print("=" * 70)
    res_base = evaluate_model(base_cols, "Baseline (Consensus Core)")
    base_acc_vh = res_base["val_hard"]["acc"]
    base_acc_va = res_base["val_all"]["acc"]
    print(f"  val_hard:   {base_acc_vh*100:.4f}% ({res_base['val_hard']['correct']}/{n_val_hard})")
    print(f"  val_all:    {base_acc_va*100:.4f}% ({res_base['val_all']['correct']}/{n_val_all})")
    print(f"  train_hard: {res_base['train_hard']['acc']*100:.4f}% ({res_base['train_hard']['correct']}/{n_train})")

    # 6. Run 1: Baseline + FEAT-G02 Surface
    print("\n" + "=" * 70)
    print("RUN 1: BASELINE + FEAT-G02 SURFACE HEURISTICS (4 + 13 = 17 features)")
    print("=" * 70)
    cols_run1 = base_cols + feat_g02_cols
    res_run1 = evaluate_model(cols_run1, "Baseline + FEAT-G02 (Surface)")
    diff_vh_run1 = res_run1["val_hard"]["acc"] - base_acc_vh
    diff_va_run1 = res_run1["val_all"]["acc"] - base_acc_va
    print(f"  val_hard:   {res_run1['val_hard']['acc']*100:.4f}% ({res_run1['val_hard']['correct']}/{n_val_hard}) -> Delta = {diff_vh_run1*100:+.4f}% ({res_run1['val_hard']['correct'] - res_base['val_hard']['correct']:+d} questions)")
    print(f"  val_all:    {res_run1['val_all']['acc']*100:.4f}% ({res_run1['val_all']['correct']}/{n_val_all}) -> Delta = {diff_va_run1*100:+.4f}%")
    print(f"  train_hard: {res_run1['train_hard']['acc']*100:.4f}% ({res_run1['train_hard']['correct']}/{n_train})")

    # 7. Run 2: Baseline + FEAT-G03 Symbolic & Reasoning
    print("\n" + "=" * 70)
    print("RUN 2: BASELINE + FEAT-G03 SYMBOLIC & REASONING (4 + 12 = 16 features)")
    print("=" * 70)
    cols_run2 = base_cols + feat_g03_cols
    res_run2 = evaluate_model(cols_run2, "Baseline + FEAT-G03 (Symbolic)")
    diff_vh_run2 = res_run2["val_hard"]["acc"] - base_acc_vh
    diff_va_run2 = res_run2["val_all"]["acc"] - base_acc_va
    print(f"  val_hard:   {res_run2['val_hard']['acc']*100:.4f}% ({res_run2['val_hard']['correct']}/{n_val_hard}) -> Delta = {diff_vh_run2*100:+.4f}% ({res_run2['val_hard']['correct'] - res_base['val_hard']['correct']:+d} questions)")
    print(f"  val_all:    {res_run2['val_all']['acc']*100:.4f}% ({res_run2['val_all']['correct']}/{n_val_all}) -> Delta = {diff_va_run2*100:+.4f}%")
    print(f"  train_hard: {res_run2['train_hard']['acc']*100:.4f}% ({res_run2['train_hard']['correct']}/{n_train})")

    # Also check FEAT-G03 without contingency column (11 features)
    cols_run2_reg = base_cols + feat_g03_reg_cols
    res_run2_reg = evaluate_model(cols_run2_reg, "Baseline + FEAT-G03 (Regression-Safe, 11 cols)")
    diff_vh_run2_reg = res_run2_reg["val_hard"]["acc"] - base_acc_vh
    diff_va_run2_reg = res_run2_reg["val_all"]["acc"] - base_acc_va
    print(f"  [Sub-Check without contingency col] val_hard: {res_run2_reg['val_hard']['acc']*100:.4f}% (Delta = {diff_vh_run2_reg*100:+.4f}%)")

    # 8. Run 3: Baseline + FEAT-G04 Geometry
    print("\n" + "=" * 70)
    print("RUN 3: BASELINE + FEAT-G04 GEOMETRY (4 + 4 = 8 features)")
    print("=" * 70)
    cols_run3 = base_cols + feat_g04_cols
    res_run3 = evaluate_model(cols_run3, "Baseline + FEAT-G04 (Geometry)")
    diff_vh_run3 = res_run3["val_hard"]["acc"] - base_acc_vh
    diff_va_run3 = res_run3["val_all"]["acc"] - base_acc_va
    print(f"  val_hard:   {res_run3['val_hard']['acc']*100:.4f}% ({res_run3['val_hard']['correct']}/{n_val_hard}) -> Delta = {diff_vh_run3*100:+.4f}% ({res_run3['val_hard']['correct'] - res_base['val_hard']['correct']:+d} questions)")
    print(f"  val_all:    {res_run3['val_all']['acc']*100:.4f}% ({res_run3['val_all']['correct']}/{n_val_all}) -> Delta = {diff_va_run3*100:+.4f}%")
    print(f"  train_hard: {res_run3['train_hard']['acc']*100:.4f}% ({res_run3['train_hard']['correct']}/{n_train})")

    # 9. Run 4: Baseline + GSM8K Answer Morphology
    print("\n" + "=" * 70)
    print("RUN 4: BASELINE + GSM8K ANSWER MORPHOLOGY (4 + 5 = 9 features)")
    print("=" * 70)
    cols_run4 = base_cols + morphology_cols
    res_run4 = evaluate_model(cols_run4, "Baseline + Morphology (5 dims)")
    diff_vh_run4 = res_run4["val_hard"]["acc"] - base_acc_vh
    diff_va_run4 = res_run4["val_all"]["acc"] - base_acc_va
    print(f"  val_hard:   {res_run4['val_hard']['acc']*100:.4f}% ({res_run4['val_hard']['correct']}/{n_val_hard}) -> Delta = {diff_vh_run4*100:+.4f}% ({res_run4['val_hard']['correct'] - res_base['val_hard']['correct']:+d} questions)")
    print(f"  val_all:    {res_run4['val_all']['acc']*100:.4f}% ({res_run4['val_all']['correct']}/{n_val_all}) -> Delta = {diff_va_run4*100:+.4f}%")
    print(f"  train_hard: {res_run4['train_hard']['acc']*100:.4f}% ({res_run4['train_hard']['correct']}/{n_train})")

    # 10. Gating Decision Analysis
    print("\n" + "=" * 70)
    print("GATE EVALUATION SUMMARY (Threshold: Delta(val_hard) >= +1.0% AND val_all not dropping)")
    print("=" * 70)
    runs_summary = [
        {"name": "FEAT-G02 (Surface)", "res": res_run1, "delta_vh": diff_vh_run1, "delta_va": diff_va_run1, "is_morphology": False},
        {"name": "FEAT-G03 (Symbolic)", "res": res_run2, "delta_vh": diff_vh_run2, "delta_va": diff_va_run2, "is_morphology": False},
        {"name": "FEAT-G04 (Geometry)", "res": res_run3, "delta_vh": diff_vh_run3, "delta_va": diff_va_run3, "is_morphology": False},
        {"name": "GSM8K Morphology", "res": res_run4, "delta_vh": diff_vh_run4, "delta_va": diff_va_run4, "is_morphology": True},
    ]

    qualified_families = []
    gating_records = []

    for item in runs_summary:
        d_vh = item["delta_vh"]
        d_va = item["delta_va"]
        acc_vh = item["res"]["val_hard"]["acc"]
        acc_va = item["res"]["val_all"]["acc"]

        meets_hard_threshold = (d_vh >= 0.0100 - 1e-6)
        val_all_healthy = (d_va >= -0.005)

        status = ""
        action_decision = ""
        if item["is_morphology"]:
            action_decision = "RETAINED AS LOW-WEIGHT CONTRAST ONLY (Forbidden from becoming core weight per Sec 4.2)"
            status = "CONTRAST_ONLY"
        elif meets_hard_threshold and val_all_healthy:
            action_decision = "QUALIFIED FOR EXP-B09/B10 (Delta >= +1.0% and val_all stable/improved)"
            status = "QUALIFIED"
            qualified_families.append(item["name"])
        elif d_vh > 0:
            action_decision = "INSUFFICIENT GAIN (< +1.0% gain, excluded from frozen models)"
            status = "EXCLUDED_MARGINAL"
        else:
            action_decision = "DEGRADED OR NEGATIVE IMPACT (Excluded from frozen models)"
            status = "EXCLUDED_DEGRADED"

        gating_records.append({
            "family": item["name"],
            "val_hard_acc": acc_vh,
            "val_hard_delta": d_vh,
            "val_all_acc": acc_va,
            "val_all_delta": d_va,
            "status": status,
            "decision": action_decision,
        })

        print(f"Family: {item['name']:<22} | val_hard: {acc_vh*100:.2f}% (Delta: {d_vh*100:+.2f}%) | val_all: {acc_va*100:.2f}% (Delta: {d_va*100:+.2f}%)")
        print(f"  --> Decision: {action_decision}")

    results = {
        "dataset": "train_hard / val_hard / val_all",
        "baseline": res_base,
        "runs": {
            "FEAT-G02": res_run1,
            "FEAT-G03": res_run2,
            "FEAT-G03_reg_safe": res_run2_reg,
            "FEAT-G04": res_run3,
            "Morphology": res_run4,
        },
        "gating_records": gating_records,
        "qualified_families": qualified_families,
    }

    # Save to disk
    runs_dir = DEV_SRC_DIR.parent / 'runs' / 'audit_logs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_file = runs_dir / 'exp_b04_ablation_audit.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull ablation audit saved to: {out_file}")

    return results


if __name__ == '__main__':
    run_exp_b04_ablation()
