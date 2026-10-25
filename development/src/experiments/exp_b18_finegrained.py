"""EXP-B18: Fine-Grained Symbolic Chain and Cluster Evidence Feature Ablation (FEAT-G05).

Evaluates the incremental contribution of fine-grained arithmetic chain continuity,
final equation verification, Markdown heading conflict detection, and answer cluster collective evidence.
Directly addresses Research Question 1 (RQ1) and Research Question 2 (RQ2).

Follows EXP-B04 incremental ablation protocol:
  - Standardized L2 Logistic Regression (C=1.0, solver='lbfgs', max_iter=1000)
  - Fits on train_hard (2,114 questions), evaluates on val_hard (355 questions) and val_all (798 questions)
  - Evaluates LightGBM decision tree ranking (16 cols vs 22 cols)
  - Verifies synergy with POST-P05 / POST-P06 postprocessing

Gate Retention Rules:
  - Rule 1: Delta(val_hard) >= +1.0 percentage points (~4 questions out of 355)
  - Rule 2: val_all is non-decreasing (Delta(val_all) >= 0.0)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

from common_utils import (
    build_label_matrix,
    get_hard_subset_mask,
    load_dataset,
    load_gold_references,
)
from features.consensus import ConsensusFeatureExtractor
from features.finegrained import FineGrainedVerificationFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"


def accuracy_dict(hits: np.ndarray) -> Dict[str, Any]:
    n = int(hits.size)
    c = int(hits.sum())
    return {"correct": c, "total": n, "accuracy": 100.0 * c / max(1, n)}


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
    scorer = OFFICIAL_CODE_DIR / "score_predictions.py"
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


def run_exp_b18() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=" * 75)
    print("Executing EXP-B18: Fine-Grained Symbolic & Cluster Evidence Ablation (FEAT-G05)")
    print("=" * 75)

    # 1. Load Column Contract
    contract_path = DEV_SRC_DIR.parent / "runs" / "audit_logs" / "private_configuration_omitted.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")
    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    # 2. Load Datasets
    print("\n--- Step 1: Loading Datasets & Splitting Subsets ---")
    bank_train, arrays_train = load_dataset("train")
    gold_train = load_gold_references("train")
    train_hard_mask = get_hard_subset_mask(bank_train)
    bank_train_hard = [b for i, b in enumerate(bank_train) if train_hard_mask[i]]
    arrays_train_hard = {"embeddings": arrays_train["embeddings"][train_hard_mask]}
    labels_train_hard = build_label_matrix(bank_train_hard, [g for i, g in enumerate(gold_train) if train_hard_mask[i]])

    bank_val, arrays_val = load_dataset("validation")
    gold_val = load_gold_references("validation")
    val_hard_mask = get_hard_subset_mask(bank_val)
    bank_val_hard = [b for i, b in enumerate(bank_val) if val_hard_mask[i]]
    arrays_val_hard = {"embeddings": arrays_val["embeddings"][val_hard_mask]}
    labels_val_hard = build_label_matrix(bank_val_hard, [g for i, g in enumerate(gold_val) if val_hard_mask[i]])
    labels_val_all = build_label_matrix(bank_val, gold_val)

    n_tr = len(bank_train_hard)
    n_vh = len(bank_val_hard)
    n_va = len(bank_val)
    print(f"  train_hard: {n_tr} questions ({n_tr * 4} candidates)")
    print(f"  val_hard:   {n_vh} questions ({n_vh * 4} candidates)")
    print(f"  val_all:    {n_va} questions ({n_va * 4} candidates)")

    # 3. Feature Extraction
    print("\n--- Step 2: Extracting Feature Families ---")
    t_feat = time.perf_counter()
    ext_g01 = ConsensusFeatureExtractor()
    ext_g03 = SymbolicVerificationFeatureExtractor()
    ext_g05 = FineGrainedVerificationFeatureExtractor()

    # Consensus Core (4 cols)
    core_cols = ["is_valid", "cluster_ratio", "is_unique_majority", "is_tied_majority"]
    core_idx = [ext_g01.FEATURE_NAMES.index(c) for c in core_cols]

    # G03 (12 cols from contract)
    g03_cols = contract["statistically_eligible_by_family"]["FEAT-G03"]
    g03_idx = [ext_g03.FEATURE_NAMES.index(c) for c in g03_cols]

    # G05 (6 cols)
    g05_cols = ext_g05.FEATURE_NAMES

    print("  Extracting train_hard features...")
    x_tr_g01_all = ext_g01.transform(bank_train_hard, arrays_train_hard)
    x_tr_core = x_tr_g01_all[:, :, core_idx]
    x_tr_g03_all = ext_g03.transform(bank_train_hard, arrays_train_hard)
    x_tr_g03 = x_tr_g03_all[:, :, g03_idx]
    x_tr_g05 = ext_g05.transform(bank_train_hard, arrays_train_hard)

    print("  Extracting val_hard features...")
    x_vh_g01_all = ext_g01.transform(bank_val_hard, arrays_val_hard)
    x_vh_core = x_vh_g01_all[:, :, core_idx]
    x_vh_g03_all = ext_g03.transform(bank_val_hard, arrays_val_hard)
    x_vh_g03 = x_vh_g03_all[:, :, g03_idx]
    x_vh_g05 = ext_g05.transform(bank_val_hard, arrays_val_hard)

    print("  Extracting val_all features...")
    x_va_g01_all = ext_g01.transform(bank_val, arrays_val)
    x_va_core = x_va_g01_all[:, :, core_idx]
    x_va_g03_all = ext_g03.transform(bank_val, arrays_val)
    x_va_g03 = x_va_g03_all[:, :, g03_idx]
    x_va_g05 = ext_g05.transform(bank_val, arrays_val)
    print(f"  Feature extraction completed in {time.perf_counter() - t_feat:.2f}s.")

    y_tr_flat = labels_train_hard.flatten()

    def evaluate_linear_model(
        x_tr: np.ndarray,
        x_vh: np.ndarray,
        x_va: np.ndarray,
        col_names: List[str],
        name: str,
    ) -> Dict[str, Any]:
        dim = len(col_names)
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_tr.reshape(-1, dim))
        x_vh_scaled = scaler.transform(x_vh.reshape(-1, dim))
        x_va_scaled = scaler.transform(x_va.reshape(-1, dim))

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(x_tr_scaled, y_tr_flat)

        logits_vh = clf.decision_function(x_vh_scaled).reshape(-1, 4)
        preds_vh = np.argmax(logits_vh, axis=1)
        hits_vh = labels_val_hard[np.arange(n_vh), preds_vh]
        acc_vh = accuracy_dict(hits_vh)

        logits_va = clf.decision_function(x_va_scaled).reshape(-1, 4)
        preds_va = np.argmax(logits_va, axis=1)
        hits_va = labels_val_all[np.arange(n_va), preds_va]
        acc_va = accuracy_dict(hits_va)

        coef_map = {col_names[j]: float(clf.coef_[0, j]) for j in range(dim)}

        return {
            "name": name,
            "feature_dim": dim,
            "features": col_names,
            "val_hard": acc_vh,
            "val_all": acc_va,
            "coefficients": coef_map,
            "intercept": float(clf.intercept_[0]),
            "_logits_va": logits_va,
            "_preds_va": preds_va,
            "_preds_vh": preds_vh,
        }

    # 4. Standardized Ablation Matrix
    print("\n--- Step 3: Standardized Incremental Ablation Matrix ---")

    # Run 0: Baseline Consensus Core (4 cols)
    print("\n[RUN 0: Baseline Consensus Core (4 features)]")
    run0 = evaluate_linear_model(
        x_tr_core, x_vh_core, x_va_core, core_cols, "Run 0 (Consensus Core, 4 cols)"
    )
    print(f"  val_hard: {run0['val_hard']['correct']}/{n_vh} ({run0['val_hard']['accuracy']:.2f}%)")
    print(f"  val_all:  {run0['val_all']['correct']}/{n_va} ({run0['val_all']['accuracy']:.2f}%)")

    # Run 1: Consensus Core + FEAT-G05 (10 cols)
    print("\n[RUN 1: Consensus Core + FEAT-G05 (10 features)]")
    cols_run1 = core_cols + g05_cols
    x_tr_run1 = np.concatenate([x_tr_core, x_tr_g05], axis=-1)
    x_vh_run1 = np.concatenate([x_vh_core, x_vh_g05], axis=-1)
    x_va_run1 = np.concatenate([x_va_core, x_va_g05], axis=-1)
    run1 = evaluate_linear_model(
        x_tr_run1, x_vh_run1, x_va_run1, cols_run1, "Run 1 (Core + FEAT-G05, 10 cols)"
    )
    d_vh_run1 = run1["val_hard"]["accuracy"] - run0["val_hard"]["accuracy"]
    d_va_run1 = run1["val_all"]["accuracy"] - run0["val_all"]["accuracy"]
    q_vh_run1 = run1["val_hard"]["correct"] - run0["val_hard"]["correct"]
    q_va_run1 = run1["val_all"]["correct"] - run0["val_all"]["correct"]
    print(f"  val_hard: {run1['val_hard']['correct']}/{n_vh} ({run1['val_hard']['accuracy']:.2f}%) -> Delta = {d_vh_run1:+.2f}% ({q_vh_run1:+d} q)")
    print(f"  val_all:  {run1['val_all']['correct']}/{n_va} ({run1['val_all']['accuracy']:.2f}%) -> Delta = {d_va_run1:+.2f}% ({q_va_run1:+d} q)")

    # Run 2: Locked 16-Col Reasoning Baseline (Core + FEAT-G03)
    print("\n[RUN 2: Locked 16-Col Reasoning Baseline (Core + FEAT-G03)]")
    cols_run2 = core_cols + g03_cols
    x_tr_run2 = np.concatenate([x_tr_core, x_tr_g03], axis=-1)
    x_vh_run2 = np.concatenate([x_vh_core, x_vh_g03], axis=-1)
    x_va_run2 = np.concatenate([x_va_core, x_va_g03], axis=-1)
    run2 = evaluate_linear_model(
        x_tr_run2, x_vh_run2, x_va_run2, cols_run2, "Run 2 (Core + FEAT-G03, 16 cols)"
    )
    print(f"  val_hard: {run2['val_hard']['correct']}/{n_vh} ({run2['val_hard']['accuracy']:.2f}%)")
    print(f"  val_all:  {run2['val_all']['correct']}/{n_va} ({run2['val_all']['accuracy']:.2f}%)")

    # Run 3: Full Stack (Core + FEAT-G03 + FEAT-G05, 22 cols)
    print("\n[RUN 3: Full Stack Reasoning & Verification (Core + G03 + G05, 22 cols)]")
    cols_run3 = core_cols + g03_cols + g05_cols
    x_tr_run3 = np.concatenate([x_tr_core, x_tr_g03, x_tr_g05], axis=-1)
    x_vh_run3 = np.concatenate([x_vh_core, x_vh_g03, x_vh_g05], axis=-1)
    x_va_run3 = np.concatenate([x_va_core, x_va_g03, x_va_g05], axis=-1)
    run3 = evaluate_linear_model(
        x_tr_run3, x_vh_run3, x_va_run3, cols_run3, "Run 3 (Core + G03 + FEAT-G05, 22 cols)"
    )
    d_vh_run3 = run3["val_hard"]["accuracy"] - run2["val_hard"]["accuracy"]
    d_va_run3 = run3["val_all"]["accuracy"] - run2["val_all"]["accuracy"]
    q_vh_run3 = run3["val_hard"]["correct"] - run2["val_hard"]["correct"]
    q_va_run3 = run3["val_all"]["correct"] - run2["val_all"]["correct"]
    print(f"  val_hard: {run3['val_hard']['correct']}/{n_vh} ({run3['val_hard']['accuracy']:.2f}%) -> Delta = {d_vh_run3:+.2f}% ({q_vh_run3:+d} q)")
    print(f"  val_all:  {run3['val_all']['correct']}/{n_va} ({run3['val_all']['accuracy']:.2f}%) -> Delta = {d_va_run3:+.2f}% ({q_va_run3:+d} q)")

    # 5. Tree Model Comparison (GBDT LightGBM: 16 cols vs 22 cols)
    print("\n--- Step 4: GBDT Comparison (LightGBM on 16 cols vs 22 cols) ---")
    lgb_16 = LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_16.fit(x_tr_run2.reshape(-1, 16), y_tr_flat)
    lgb_vh_16 = np.argmax(lgb_16.predict_proba(x_vh_run2.reshape(-1, 16))[:, 1].reshape(-1, 4), axis=1)
    lgb_va_16 = np.argmax(lgb_16.predict_proba(x_va_run2.reshape(-1, 16))[:, 1].reshape(-1, 4), axis=1)
    acc_lgb_vh_16 = accuracy_dict(labels_val_hard[np.arange(n_vh), lgb_vh_16])
    acc_lgb_va_16 = accuracy_dict(labels_val_all[np.arange(n_va), lgb_va_16])

    lgb_22 = LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_22.fit(x_tr_run3.reshape(-1, 22), y_tr_flat)
    lgb_vh_22 = np.argmax(lgb_22.predict_proba(x_vh_run3.reshape(-1, 22))[:, 1].reshape(-1, 4), axis=1)
    lgb_va_22 = np.argmax(lgb_22.predict_proba(x_va_run3.reshape(-1, 22))[:, 1].reshape(-1, 4), axis=1)
    acc_lgb_vh_22 = accuracy_dict(labels_val_hard[np.arange(n_vh), lgb_vh_22])
    acc_lgb_va_22 = accuracy_dict(labels_val_all[np.arange(n_va), lgb_va_22])

    print(f"  LightGBM (16 cols): val_hard = {acc_lgb_vh_16['correct']}/{n_vh} ({acc_lgb_vh_16['accuracy']:.2f}%), val_all = {acc_lgb_va_16['correct']}/{n_va} ({acc_lgb_va_16['accuracy']:.2f}%)")
    print(f"  LightGBM (22 cols): val_hard = {acc_lgb_vh_22['correct']}/{n_vh} ({acc_lgb_vh_22['accuracy']:.2f}%), val_all = {acc_lgb_va_22['correct']}/{n_va} ({acc_lgb_va_22['accuracy']:.2f}%)")
    print(f"  Incremental GBDT (16 -> 22): val_hard {acc_lgb_vh_22['correct'] - acc_lgb_vh_16['correct']:+d} q, val_all {acc_lgb_va_22['correct'] - acc_lgb_va_16['correct']:+d} q")

    # 6. Postprocessing Synergy Test (POST-P05 Validity Masking + Heading Conflict De-mining)
    print("\n--- Step 5: Postprocessing Synergy Test on Run 3 (22 cols) ---")
    logits_va_run3 = run3["_logits_va"].copy()
    masked_va_m1 = apply_validity_mask(logits_va_run3, bank_val)
    masked_va_m2 = apply_conflict_penalty(masked_va_m1, bank_val)

    preds_m2 = np.argmax(masked_va_m2, axis=1)
    hits_m2_va = labels_val_all[np.arange(n_va), preds_m2]
    hits_m2_vh = labels_val_hard[np.arange(n_vh), preds_m2[val_hard_mask]]
    acc_m2_va = accuracy_dict(hits_m2_va)
    acc_m2_vh = accuracy_dict(hits_m2_vh)
    print(f"  Run 3 Raw:        val_hard = {run3['val_hard']['correct']}/{n_vh}, val_all = {run3['val_all']['correct']}/{n_va}")
    print(f"  Run 3 + POST-P05: val_hard = {acc_m2_vh['correct']}/{n_vh} ({acc_m2_vh['accuracy']:.2f}%), val_all = {acc_m2_va['correct']}/{n_va} ({acc_m2_va['accuracy']:.2f}%)")
    print(f"  POST-P05 Gain:    val_hard {acc_m2_vh['correct'] - run3['val_hard']['correct']:+d} q, val_all {acc_m2_va['correct'] - run3['val_all']['correct']:+d} q (+{acc_m2_va['correct'] - run3['val_all']['correct']} easy questions saved)")

    # 7. Gate Rule Verification
    print("\n--- Step 6: Gate Rule Verification ---")
    gate_rule1 = bool(d_vh_run1 >= 1.0)
    gate_rule2 = bool(d_va_run1 >= 0.0)
    gate_rule1_stack = bool(d_vh_run3 >= 1.0)
    gate_rule2_stack = bool(d_va_run3 >= 0.0)
    passed = gate_rule1 and gate_rule2 and gate_rule1_stack and gate_rule2_stack
    decision = "RETAINED" if passed else "REJECTED"
    print(f"  Rule 1 (Delta val_hard >= +1.0% over Core):   {d_vh_run1:+.2f}% -> {'PASSED' if gate_rule1 else 'FAILED'}")
    print(f"  Rule 2 (Delta val_all >= 0.0 over Core):       {d_va_run1:+.2f}% -> {'PASSED' if gate_rule2 else 'FAILED'}")
    print(f"  Rule 1b (Delta val_hard >= +1.0% over 16col):  {d_vh_run3:+.2f}% -> {'PASSED' if gate_rule1_stack else 'FAILED'}")
    print(f"  Rule 2b (Delta val_all >= 0.0 over 16col):     {d_va_run3:+.2f}% -> {'PASSED' if gate_rule2_stack else 'FAILED'}")
    print(f"  ==> Overall Retention Decision: {decision}")

    # 8. Official Predictions and Scoring
    print("\n--- Step 7: Exporting Official Predictions and Running Scorer ---")
    pred_dir = DEV_SRC_DIR.parent / "runs" / "predictions"
    score_dir = DEV_SRC_DIR.parent / "runs" / "scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    official_scores: Dict[str, Any] = {}

    configs_to_score = [
        ("core_plus_g05", run1["_preds_va"], "EXP-B18_core_plus_g05"),
        ("full_stack_22col", run3["_preds_va"], "EXP-B18_full_stack_22col"),
        ("full_stack_22col_post", preds_m2, "EXP-B18_full_stack_22col_post"),
    ]

    for key, chosen_idxs, prefix in configs_to_score:
        p_path = pred_dir / f"{prefix}_predictions.json"
        s_path = score_dir / f"{prefix}_score.json"
        write_official_predictions(p_path, bank_val, chosen_idxs)
        sc = official_score(p_path, s_path)
        official_scores[key] = sc
        print(f"  Official Scorer [{prefix}]: {sc['correct']} / {sc['n']} ({sc['accuracy']:.2f}%)")

    # Clean internal arrays before audit serialization
    for r in [run0, run1, run2, run3]:
        r.pop("_logits_va", None)
        r.pop("_preds_va", None)
        r.pop("_preds_vh", None)

    # 9. Audit Logging
    audit_data = {
        "experiment": "EXP-B18",
        "description": "Fine-Grained Symbolic Chain and Cluster Evidence Feature Extractor Ablation (FEAT-G05)",
        "protocol": PROTOCOL,
        "execution_seconds": round(time.perf_counter() - t0, 2),
        "feature_definitions": {
            "final_eq_matches_answer": "1.0 if RHS of last verified equation matches parsed answer, else 0.0",
            "equation_chain_continuity": "fraction of operands in eq k>=2 originating from question stem or prior RHS",
            "has_parser_heading_conflict": "1.0 if candidate contains Markdown heading conflict trap, else 0.0",
            "cluster_avg_correct_eq_ratio": "mean correct equation ratio across candidates in same answer cluster",
            "cluster_max_condition_coverage": "max question condition coverage across candidates in same answer cluster",
            "cluster_min_wrong_eq": "min wrong equation count across candidates in same answer cluster",
        },
        "dataset_splits": {
            "train_hard_questions": n_tr,
            "val_hard_questions": n_vh,
            "val_all_questions": n_va,
        },
        "ablation_matrix": {
            "run_0_consensus_core_4col": run0,
            "run_1_core_plus_feat_g05_10col": run1,
            "run_2_core_plus_feat_g03_16col": run2,
            "run_3_full_stack_22col": run3,
            "incremental_deltas": {
                "run1_vs_run0_val_hard_delta_pct": d_vh_run1,
                "run1_vs_run0_val_hard_delta_questions": q_vh_run1,
                "run1_vs_run0_val_all_delta_pct": d_va_run1,
                "run1_vs_run0_val_all_delta_questions": q_va_run1,
                "run3_vs_run2_val_hard_delta_pct": d_vh_run3,
                "run3_vs_run2_val_hard_delta_questions": q_vh_run3,
                "run3_vs_run2_val_all_delta_pct": d_va_run3,
                "run3_vs_run2_val_all_delta_questions": q_va_run3,
            },
        },
        "gbdt_comparison": {
            "lightgbm_16col": {"val_hard": acc_lgb_vh_16, "val_all": acc_lgb_va_16},
            "lightgbm_22col": {"val_hard": acc_lgb_vh_22, "val_all": acc_lgb_va_22},
        },
        "postprocessing_synergy": {
            "run_3_with_post_p05": {
                "val_hard": acc_m2_vh,
                "val_all": acc_m2_va,
            }
        },
        "gate_verification": {
            "rule_1_run1_val_hard_ge_1pct": gate_rule1,
            "rule_2_run1_val_all_ge_0pct": gate_rule2,
            "rule_1b_run3_val_hard_ge_1pct": gate_rule1_stack,
            "rule_2b_run3_val_all_ge_0pct": gate_rule2_stack,
            "decision": decision,
        },
        "official_scores": official_scores,
    }

    audit_path = DEV_SRC_DIR.parent / "runs" / "audit_logs" / "exp_b18_finegrained_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAudit log saved to: {audit_path}")
    print("=" * 75)
    return audit_data


if __name__ == "__main__":
    run_exp_b18()
