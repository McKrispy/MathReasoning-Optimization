"""EXP-B05: Surface Features Alone & False Positive Analysis (Responding to RQ2).

Investigates what happens when selecting candidates purely based on surface formatting,
length, and confident wording (FEAT-G02) without considering consensus or semantic logic.

1. Reads eligible columns from dev/runs/audit_logs/private_configuration_omitted.json.
2. Trains MOD-M04 (HistGradientBoostingClassifier / Decision Tree Reranker) on FEAT-G02.
3. Evaluates greedy candidate selection on val_hard (355 questions) and val_all (798 questions).
4. Quantifies False Positive Cases:
   - Error rate of surface-based selection.
   - Surface bias analysis (length bias, confidence bias).
   - Diagnostic cross-validation with FEAT-G03 (arithmetic calculation errors & missing conditions).
   - Detailed qualitative case studies of persuasive but incorrect candidates.
5. Saves all audit artifacts to dev/runs/audit_logs/, predictions to dev/runs/predictions/,
   and scores to dev/runs/scores/.
"""
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure dev/src is in sys.path
DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import (
    load_dataset,
    load_gold_references,
    get_hard_subset_mask,
    build_label_matrix,
)
from features.surface import SurfaceFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor


def run_exp_b05() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B05: Surface Features Alone & False Positive Analysis (RQ2)")
    print("=" * 70)

    # 1. Load column contract strictly from JSON
    contract_path = DEV_SRC_DIR.parent / 'runs' / 'audit_logs' / 'private_configuration_omitted.json'
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract JSON not found: {contract_path}")

    with open(contract_path, 'r', encoding='utf-8') as f:
        contract = json.load(f)

    surface_cols = contract["statistically_eligible_by_family"]["FEAT-G02"]
    print(f"Loaded surface feature columns ({len(surface_cols)} dims) from contract:")
    print(f"  {surface_cols}")

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

    n_train_hard = len(bank_train_hard)
    n_val_hard = len(bank_val_hard)
    n_val_all = len(bank_val)
    print(f"Loaded train_hard: {n_train_hard} questions ({n_train_hard*4} candidates)")
    print(f"Loaded val_hard:   {n_val_hard} questions ({n_val_hard*4} candidates)")
    print(f"Loaded val_all:    {n_val_all} questions ({n_val_all*4} candidates)")

    # 3. Extract FEAT-G02 Surface features
    ext_surface = SurfaceFeatureExtractor()
    feat2idx_surface = {name: i for i, name in enumerate(ext_surface.FEATURE_NAMES)}
    surface_indices = [feat2idx_surface[c] for c in surface_cols]

    print("\nExtracting FEAT-G02 surface features...")
    x_tr_surface_raw = ext_surface.transform(bank_train_hard, arrays_train_hard)  # [N, 4, 16]
    x_vh_surface_raw = ext_surface.transform(bank_val_hard, arrays_val_hard)      # [355, 4, 16]
    x_va_surface_raw = ext_surface.transform(bank_val, arrays_val)                # [798, 4, 16]

    x_tr_surface = x_tr_surface_raw[:, :, surface_indices]  # [N, 4, 13]
    x_vh_surface = x_vh_surface_raw[:, :, surface_indices]  # [355, 4, 13]
    x_va_surface = x_va_surface_raw[:, :, surface_indices]  # [798, 4, 13]

    # Also extract FEAT-G03 on val_hard for diagnostic false positive cross-validation
    print("Extracting FEAT-G03 symbolic features for false positive diagnostics...")
    ext_symbolic = SymbolicVerificationFeatureExtractor()
    x_vh_symbolic = ext_symbolic.transform(bank_val_hard, arrays_val_hard)  # [355, 4, 14]
    sym2idx = {name: i for i, name in enumerate(ext_symbolic.FEATURE_NAMES)}

    # 4. Train MOD-M04 (HistGradientBoostingClassifier)
    print("\nTraining MOD-M04 (GBDT Reranker) on FEAT-G02...")
    x_tr_flat = x_tr_surface.reshape(n_train_hard * 4, len(surface_cols))
    y_tr_flat = labels_train_hard.reshape(-1)

    gbdt = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    gbdt.fit(x_tr_flat, y_tr_flat)

    # Measure asset size (in bytes)
    import pickle
    model_bytes = len(pickle.dumps(gbdt))
    model_size_mib = model_bytes / (1024 * 1024)
    print(f"Trained GBDT model asset size: {model_bytes} bytes ({model_size_mib:.4f} MiB)")

    # 5. Evaluate inference and latency on validation sets
    print("\nRunning inference on val_hard (355 questions)...")
    t0 = time.perf_counter()
    x_vh_flat = x_vh_surface.reshape(n_val_hard * 4, len(surface_cols))
    scores_vh = gbdt.predict_proba(x_vh_flat)[:, 1].reshape(n_val_hard, 4)
    pred_vh = np.argmax(scores_vh, axis=1)
    latency_vh = time.perf_counter() - t0

    acc_vh = float(np.mean([labels_val_hard[i, pred_vh[i]] for i in range(n_val_hard)]))
    correct_vh = int(np.sum([labels_val_hard[i, pred_vh[i]] for i in range(n_val_hard)]))

    print("\nRunning inference on val_all (798 questions)...")
    t0 = time.perf_counter()
    x_va_flat = x_va_surface.reshape(n_val_all * 4, len(surface_cols))
    scores_va = gbdt.predict_proba(x_va_flat)[:, 1].reshape(n_val_all, 4)
    pred_va = np.argmax(scores_va, axis=1)
    latency_va = time.perf_counter() - t0

    acc_va = float(np.mean([labels_val_all[i, pred_va[i]] for i in range(n_val_all)]))
    correct_va = int(np.sum([labels_val_all[i, pred_va[i]] for i in range(n_val_all)]))

    # train_hard score
    scores_tr = gbdt.predict_proba(x_tr_flat)[:, 1].reshape(n_train_hard, 4)
    pred_tr = np.argmax(scores_tr, axis=1)
    acc_tr = float(np.mean([labels_train_hard[i, pred_tr[i]] for i in range(n_train_hard)]))
    correct_tr = int(np.sum([labels_train_hard[i, pred_tr[i]] for i in range(n_train_hard)]))

    print(f"\nEXP-B05 Evaluation Results:")
    print(f"  val_hard:   {acc_vh*100:.4f}% ({correct_vh} / {n_val_hard}) [Majority Voting: 69.0141%]")
    print(f"  val_all:    {acc_va*100:.4f}% ({correct_va} / {n_val_all}) [Majority Voting: 86.2155%]")
    print(f"  train_hard: {acc_tr*100:.4f}% ({correct_tr} / {n_train_hard}) [Majority Voting: 66.8398%]")
    print(f"  Inference latency (val_all): {latency_va:.4f}s")

    # 6. Quantitative False Positive Analysis (Responding to RQ2)
    print("\n" + "=" * 70)
    print("QUANTITATIVE FALSE POSITIVE ANALYSIS (Responding to RQ2)")
    print("=" * 70)

    # Identify false positive cases in val_hard:
    # Cases where the model chose a candidate with highest surface score, BUT it is incorrect (y == 0)
    false_positives = []
    true_positives = []

    for i in range(n_val_hard):
        chosen_c = int(pred_vh[i])
        is_correct = (labels_val_hard[i, chosen_c] == 1)
        item = {
            "q_idx": i,
            "q_id": bank_val_hard[i]["id"],
            "question": bank_val_hard[i]["question"],
            "chosen_candidate": chosen_c,
            "chosen_score": float(scores_vh[i, chosen_c]),
            "is_correct": is_correct,
            "chosen_surface": {
                name: float(x_vh_surface_raw[i, chosen_c, feat2idx_surface[name]])
                for name in ext_surface.FEATURE_NAMES
            },
            "chosen_symbolic": {
                name: float(x_vh_symbolic[i, chosen_c, sym2idx[name]])
                for name in ext_symbolic.FEATURE_NAMES
            },
            "chosen_text": bank_val_hard[i]["candidates"][chosen_c]["text"],
            "gold_answer": gold_val[int(np.where([g["id"] == bank_val_hard[i]["id"] for g in gold_val])[0][0])]["answer"],
        }
        if is_correct:
            true_positives.append(item)
        else:
            false_positives.append(item)

    n_fp = len(false_positives)
    n_tp = len(true_positives)
    fp_rate = n_fp / n_val_hard
    print(f"Total val_hard questions: {n_val_hard}")
    print(f"False Positives (Surface model picked WRONG candidate): {n_fp} ({fp_rate*100:.2f}%)")
    print(f"True Positives (Surface model picked CORRECT candidate): {n_tp} ({n_tp/n_val_hard*100:.2f}%)")

    # Contrast surface metrics: False Positives vs True Positives
    fp_char_len = np.mean([fp["chosen_surface"]["char_count"] for fp in false_positives])
    tp_char_len = np.mean([tp["chosen_surface"]["char_count"] for tp in true_positives])

    fp_step_count = np.mean([fp["chosen_surface"]["step_count"] for fp in false_positives])
    tp_step_count = np.mean([tp["chosen_surface"]["step_count"] for tp in true_positives])

    fp_conf_count = np.mean([fp["chosen_surface"]["confidence_count"] for fp in false_positives])
    tp_conf_count = np.mean([tp["chosen_surface"]["confidence_count"] for tp in true_positives])

    fp_math_density = np.mean([fp["chosen_surface"]["math_symbol_density"] for fp in false_positives])
    tp_math_density = np.mean([tp["chosen_surface"]["math_symbol_density"] for tp in true_positives])

    # Check how many false positives have reasoning/calculation errors according to FEAT-G03
    fp_with_arith_error = sum(1 for fp in false_positives if fp["chosen_symbolic"]["has_arithmetic_error"] > 0.5)
    fp_with_missing_cond = sum(1 for fp in false_positives if fp["chosen_symbolic"]["has_missing_condition"] > 0.5)
    fp_with_either_error = sum(1 for fp in false_positives if fp["chosen_symbolic"]["has_arithmetic_error"] > 0.5 or fp["chosen_symbolic"]["has_missing_condition"] > 0.5)

    print(f"\nSurface Feature Comparison:")
    print(f"  Average Character Length: False Positives = {fp_char_len:.2f} chars vs True Positives = {tp_char_len:.2f} chars")
    print(f"  Average Step Marker Count: False Positives = {fp_step_count:.2f} vs True Positives = {tp_step_count:.2f}")
    print(f"  Average Confidence Words:  False Positives = {fp_conf_count:.2f} vs True Positives = {tp_conf_count:.2f}")
    print(f"  Math Symbol Density:       False Positives = {fp_math_density:.4f} vs True Positives = {tp_math_density:.4f}")

    print(f"\nDiagnostic Flaws Uncovered in False Positives (via FEAT-G03):")
    print(f"  Explicit Arithmetic Error present:  {fp_with_arith_error} / {n_fp} ({fp_with_arith_error/n_fp*100:.2f}%)")
    print(f"  Missing Question Conditions:        {fp_with_missing_cond} / {n_fp} ({fp_with_missing_cond/n_fp*100:.2f}%)")
    print(f"  Flawed in either arithmetic/cond:   {fp_with_either_error} / {n_fp} ({fp_with_either_error/n_fp*100:.2f}%)")

    # 7. Select 3 Concrete Case Studies of False Positives
    case_studies = []
    # Find illustrative cases: has high confidence / length, but arithmetic error or missing condition
    candidate_cases = [fp for fp in false_positives if fp["chosen_symbolic"]["has_arithmetic_error"] > 0.5 and fp["chosen_surface"]["char_count"] > 350]
    for case in candidate_cases[:3]:
        case_studies.append({
            "question_id": case["q_id"],
            "question_snippet": case["question"][:150] + ("..." if len(case["question"]) > 150 else ""),
            "chosen_candidate_idx": case["chosen_candidate"],
            "surface_model_score": case["chosen_score"],
            "char_count": case["chosen_surface"]["char_count"],
            "confidence_count": case["chosen_surface"]["confidence_count"],
            "step_count": case["chosen_surface"]["step_count"],
            "arithmetic_errors": case["chosen_symbolic"]["num_wrong_equations"],
            "missing_conditions": case["chosen_symbolic"]["missing_condition_count"],
            "candidate_text_snippet": case["chosen_text"][:250] + ("..." if len(case["chosen_text"]) > 250 else ""),
            "gold_answer": case["gold_answer"][:80],
        })

    print(f"\nSelected {len(case_studies)} Qualitative Case Studies for Report Section 2 (RQ2).")

    # 8. Save output data files
    runs_dir = DEV_SRC_DIR.parent / 'runs'
    audit_dir = runs_dir / 'audit_logs'
    pred_dir = runs_dir / 'predictions'
    score_dir = runs_dir / 'scores'
    audit_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed audit json
    audit_results = {
        "dataset": "val_hard (355 questions) / val_all (798 questions)",
        "features": surface_cols,
        "model": "MOD-M04 (HistGradientBoostingClassifier)",
        "asset_size_bytes": model_bytes,
        "asset_size_mib": model_size_mib,
        "latency_seconds_val_all": latency_va,
        "accuracy": {
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
                "total": n_train_hard,
            },
        },
        "false_positive_analysis": {
            "total_questions": n_val_hard,
            "false_positives_count": n_fp,
            "false_positive_rate": fp_rate,
            "true_positives_count": n_tp,
            "surface_comparison": {
                "false_positive_char_len": fp_char_len,
                "true_positive_char_len": tp_char_len,
                "false_positive_step_count": fp_step_count,
                "true_positive_step_count": tp_step_count,
                "false_positive_confidence_count": fp_conf_count,
                "true_positive_confidence_count": tp_conf_count,
            },
            "symbolic_verification_of_false_positives": {
                "has_arithmetic_error_count": fp_with_arith_error,
                "has_arithmetic_error_ratio": fp_with_arith_error / n_fp if n_fp > 0 else 0,
                "has_missing_condition_count": fp_with_missing_cond,
                "has_missing_condition_ratio": fp_with_missing_cond / n_fp if n_fp > 0 else 0,
                "has_either_flaw_count": fp_with_either_error,
                "has_either_flaw_ratio": fp_with_either_error / n_fp if n_fp > 0 else 0,
            },
            "representative_case_studies": case_studies,
        },
    }

    audit_out = audit_dir / 'exp_b05_surface_audit.json'
    with open(audit_out, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print(f"Audit log saved to: {audit_out}")

    # Save predictions format
    pred_out = pred_dir / 'EXP-B05_surface_predictions.json'
    pred_data = {
        "split": "validation",
        "predictions": [int(p) for p in pred_va],
    }
    with open(pred_out, 'w', encoding='utf-8') as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to: {pred_out}")

    # Save score format
    score_out = score_dir / 'EXP-B05_surface_score.json'
    score_data = {
        "split": "validation",
        "total": n_val_all,
        "correct": correct_va,
        "accuracy": acc_va,
        "val_hard_total": n_val_hard,
        "val_hard_correct": correct_vh,
        "val_hard_accuracy": acc_vh,
    }
    with open(score_out, 'w', encoding='utf-8') as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    print(f"Score saved to: {score_out}")

    return audit_results


if __name__ == '__main__':
    run_exp_b05()
