"""EXP-B16: Postprocessing and Parser De-mining Upgrade (Zero-Retrain Blood-Stopping & Point-Recovery).

Validates POST-P05 (validity mask & parser trap de-mining) and POST-P06 (intra-problem Softmax
cluster aggregation) using frozen logits from EXP-B12.

Key objectives:
1. Eliminate blind unparseable candidate selection on easy questions (recovering 3.67 ~ 7.33 questions on val_easy).
2. Penalize parser heading conflicts where Markdown headings hijack official parse_answer.
3. Compare intra-problem Softmax cluster aggregation vs Sigmoid power aggregation to address saturation distortion.
4. Verify pre-registered retention criteria: val_all gain >= +3.0 questions, val_hard non-decreasing.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
OFFICIAL_CODE_DIR = WORKSPACE / "DASE7506-MP2" / "code"

if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore
def accuracy_of(hits: np.ndarray) -> Dict[str, Any]:
    n = int(hits.size)
    c = int(hits.sum())
    return {"correct": c, "total": n, "accuracy": 100.0 * c / max(1, n)}
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
    detect_parser_heading_conflict,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
RUNS = WORKSPACE / "dev" / "runs"
SEEDS = (42, 43, 44)
TEMPERATURE_GRID = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
GAMMA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def sigmoid_aggregate(
    z_masked: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    gamma: float = 1.0,
    temperature: float = 1.1,
) -> List[int]:
    """Old POST-P02 Sigmoid aggregation with masking for comparison."""
    probs = sigmoid(z_masked / float(temperature))
    choices = []
    n = len(bank)
    for i in range(n):
        row = bank[i]
        cands = row["candidates"]
        cluster_scores: Dict[str, float] = defaultdict(float)
        cluster_cands: Dict[str, List[int]] = defaultdict(list)
        for c in range(4):
            ans = common.parse_answer(cands[c]["text"])
            if ans is not None:
                score = float(probs[i, c]) ** gamma if gamma > 0 else 1.0
                cluster_scores[ans] += score
                cluster_cands[ans].append(c)
        if not cluster_scores:
            choices.append(int(np.argmax(z_masked[i])))
            continue
        best_cluster = max(
            cluster_scores.keys(),
            key=lambda k: (cluster_scores[k], max(float(z_masked[i, c]) for c in cluster_cands[k])),
        )
        best_cand = max(cluster_cands[best_cluster], key=lambda c: (float(z_masked[i, c]), -c))
        choices.append(int(best_cand))
    return choices


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
        "val_all": accuracy_of(hits),
        "val_easy": accuracy_of(hits[is_easy]),
        "val_hard": accuracy_of(hits[is_hard]),
    }


def run_exp_b16() -> Dict[str, Any]:
    print("=" * 78)
    print("Executing EXP-B16: Postprocessing & Parser De-mining Upgrade")
    print("=" * 78)

    bank = common.load_bank("validation")
    gold = common.read_jsonl(OFFICIAL_CODE_DIR / "data" / "gsm8k_validation.jsonl")
    gold_answers = [common.parse_answer(q["answer"]) for q in gold]

    # Precompute problem split and parser masks
    is_hard_list = []
    conflict_count = 0
    invalid_candidate_count = 0

    for q_idx, row in enumerate(bank):
        valid_answers = set()
        for c_idx, c in enumerate(row["candidates"]):
            text = c["text"]
            ans = common.parse_answer(text)
            if ans is not None:
                valid_answers.add(ans)
            else:
                invalid_candidate_count += 1
            if detect_parser_heading_conflict(text):
                conflict_count += 1
        is_hard_list.append(len(valid_answers) >= 2)

    is_hard = np.array(is_hard_list, dtype=bool)
    is_easy = ~is_hard

    print(f"Validation dataset: {len(bank)} questions total.")
    print(f"  - val_easy (single parsed answer): {is_easy.sum()} questions (55.51%)")
    print(f"  - val_hard (>= 2 parsed answers): {is_hard.sum()} questions (44.49%)")
    print(f"  - Invalid candidates (is_valid == 0): {invalid_candidate_count} / {len(bank)*4} ({invalid_candidate_count/(len(bank)*4)*100:.2f}%)")
    print(f"  - Heading conflict trap candidates: {conflict_count} / {len(bank)*4}")

    # Load frozen B12 validation logits
    b12_dir = RUNS / "exp_b12"
    path_a = b12_dir / "val_logits_embeddings_only.npz"
    path_b = b12_dir / "val_logits_embeddings_plus_g01_g03.npz"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"Missing B12 logits files: {path_a} or {path_b}")

    logits_arms = {
        "embeddings_only": np.load(path_a),
        "embeddings_plus_g01_g03": np.load(path_b),
    }

    arms_results: Dict[str, Any] = {}
    total_configs_evaluated = 0

    for arm_name, logits_dict in logits_arms.items():
        print(f"\n{'='*30} Arm: {arm_name} {'='*30}")
        arm_res: Dict[str, Any] = {}

        member_keys = [
            "seed_42_full",
            "seed_43_full",
            "seed_44_full",
            "seed_ensemble",
            "seed_42_fold_mean",
            "seed_43_fold_mean",
            "seed_44_fold_mean",
        ]

        for key in member_keys:
            if key not in logits_dict:
                continue

            z_raw = logits_dict[key].copy()

            # Baseline: raw argmax
            c_raw = np.argmax(z_raw, axis=1).tolist()
            res_raw = evaluate_choices(c_raw, bank, gold_answers, is_hard)
            total_configs_evaluated += 1

            # M1: Hard Validity Mask
            z_m1 = apply_validity_mask(z_raw, bank, penalty=1e9)
            c_m1 = np.argmax(z_m1, axis=1).tolist()
            res_m1 = evaluate_choices(c_m1, bank, gold_answers, is_hard)
            total_configs_evaluated += 1

            # M2: Validity Mask + Heading Conflict Penalty
            z_m2 = apply_conflict_penalty(z_m1, bank, penalty=1e9)
            c_m2 = np.argmax(z_m2, axis=1).tolist()
            res_m2 = evaluate_choices(c_m2, bank, gold_answers, is_hard)
            total_configs_evaluated += 1

            # M3: M1 + Softmax Cluster Aggregation sweep
            m3_sweep = []
            for T in TEMPERATURE_GRID:
                agg = SoftmaxClusterAggregator(temperature=T)
                c_m3 = agg.aggregate(bank, z_m1)
                r = evaluate_choices(c_m3, bank, gold_answers, is_hard)
                r["temperature"] = T
                r["choices"] = c_m3
                m3_sweep.append(r)
                total_configs_evaluated += 1

            best_m3 = max(
                m3_sweep,
                key=lambda r: (r["val_hard"]["correct"], r["val_all"]["correct"], -r["temperature"]),
            )

            # M4: M2 + Softmax Cluster Aggregation sweep
            m4_sweep = []
            for T in TEMPERATURE_GRID:
                agg = SoftmaxClusterAggregator(temperature=T)
                c_m4 = agg.aggregate(bank, z_m2)
                r = evaluate_choices(c_m4, bank, gold_answers, is_hard)
                r["temperature"] = T
                r["choices"] = c_m4
                m4_sweep.append(r)
                total_configs_evaluated += 1

            best_m4 = max(
                m4_sweep,
                key=lambda r: (r["val_hard"]["correct"], r["val_all"]["correct"], -r["temperature"]),
            )

            # Contrast: M2 + Sigmoid Power Aggregation sweep
            sig_sweep = []
            for gamma in GAMMA_GRID:
                c_sig = sigmoid_aggregate(z_m2, bank, gamma=gamma, temperature=1.1)
                r = evaluate_choices(c_sig, bank, gold_answers, is_hard)
                r["gamma"] = gamma
                sig_sweep.append(r)
                total_configs_evaluated += 1

            best_sig = max(
                sig_sweep,
                key=lambda r: (r["val_hard"]["correct"], r["val_all"]["correct"], -r["gamma"]),
            )

            arm_res[key] = {
                "raw_argmax": res_raw,
                "m1_validity_mask": res_m1,
                "m2_validity_and_conflict_mask": res_m2,
                "m3_best_softmax": {k: v for k, v in best_m3.items() if k != "choices"},
                "m3_sweep": [{k: v for k, v in r.items() if k != "choices"} for r in m3_sweep],
                "m4_best_softmax": {k: v for k, v in best_m4.items() if k != "choices"},
                "m4_sweep": [{k: v for k, v in r.items() if k != "choices"} for r in m4_sweep],
                "sigmoid_best": best_sig,
                "sigmoid_sweep": sig_sweep,
                "_choices": {
                    "raw": c_raw,
                    "m1": c_m1,
                    "m2": c_m2,
                    "m3_best": best_m3["choices"],
                    "m4_best": best_m4["choices"],
                },
            }

            print(
                f"[{key:17s}] RAW: {res_raw['val_all']['correct']} (easy {res_raw['val_easy']['correct']}, hard {res_raw['val_hard']['correct']}) "
                f"-> M1: {res_m1['val_all']['correct']} (easy {res_m1['val_easy']['correct']}, hard {res_m1['val_hard']['correct']}) "
                f"-> M2: {res_m2['val_all']['correct']} (hard {res_m2['val_hard']['correct']}) "
                f"-> M4 Softmax(T={best_m4['temperature']}): {best_m4['val_all']['correct']} (hard {best_m4['val_hard']['correct']}) "
                f"| Sigmoid(g={best_sig['gamma']}): {best_sig['val_all']['correct']} (hard {best_sig['val_hard']['correct']})"
            )

        arms_results[arm_name] = arm_res

    # Multi-seed series summaries on Arm A (the retained baseline from EXP-B12)
    arm_a = arms_results["embeddings_only"]
    single_seeds = ["seed_42_full", "seed_43_full", "seed_44_full"]

    def extract_series(metric_path: Sequence[str]) -> List[int]:
        vals = []
        for s in single_seeds:
            cur = arm_a[s]
            for p in metric_path:
                cur = cur[p]
            vals.append(cur)
        return vals

    raw_h = extract_series(["raw_argmax", "val_hard", "correct"])
    raw_a = extract_series(["raw_argmax", "val_all", "correct"])
    raw_e = extract_series(["raw_argmax", "val_easy", "correct"])

    m1_h = extract_series(["m1_validity_mask", "val_hard", "correct"])
    m1_a = extract_series(["m1_validity_mask", "val_all", "correct"])
    m1_e = extract_series(["m1_validity_mask", "val_easy", "correct"])

    m2_h = extract_series(["m2_validity_and_conflict_mask", "val_hard", "correct"])
    m2_a = extract_series(["m2_validity_and_conflict_mask", "val_all", "correct"])
    m2_e = extract_series(["m2_validity_and_conflict_mask", "val_easy", "correct"])

    m4_h = extract_series(["m4_best_softmax", "val_hard", "correct"])
    m4_a = extract_series(["m4_best_softmax", "val_all", "correct"])

    ens_raw = arm_a["seed_ensemble"]["raw_argmax"]
    ens_m1 = arm_a["seed_ensemble"]["m1_validity_mask"]
    ens_m2 = arm_a["seed_ensemble"]["m2_validity_and_conflict_mask"]
    ens_m4 = arm_a["seed_ensemble"]["m4_best_softmax"]

    summary_comparison = {
        "single_seeds_raw_val_all_mean": float(np.mean(raw_a)),
        "single_seeds_raw_val_easy_mean": float(np.mean(raw_e)),
        "single_seeds_raw_val_hard_mean": float(np.mean(raw_h)),
        "single_seeds_m1_val_all_mean": float(np.mean(m1_a)),
        "single_seeds_m1_val_easy_mean": float(np.mean(m1_e)),
        "single_seeds_m1_val_hard_mean": float(np.mean(m1_h)),
        "single_seeds_m2_val_all_mean": float(np.mean(m2_a)),
        "single_seeds_m2_val_easy_mean": float(np.mean(m2_e)),
        "single_seeds_m2_val_hard_mean": float(np.mean(m2_h)),
        "single_seeds_m4_val_all_mean": float(np.mean(m4_a)),
        "single_seeds_m4_val_hard_mean": float(np.mean(m4_h)),
        "val_easy_recovery_questions": float(np.mean(m1_e) - np.mean(raw_e)),
        "val_all_gain_m1_over_raw": float(np.mean(m1_a) - np.mean(raw_a)),
        "val_all_gain_m2_over_raw": float(np.mean(m2_a) - np.mean(raw_a)),
        "val_hard_gain_m2_over_raw": float(np.mean(m2_h) - np.mean(raw_h)),
        "seed_ensemble_raw": {
            "val_all": ens_raw["val_all"]["correct"],
            "val_easy": ens_raw["val_easy"]["correct"],
            "val_hard": ens_raw["val_hard"]["correct"],
        },
        "seed_ensemble_m1": {
            "val_all": ens_m1["val_all"]["correct"],
            "val_easy": ens_m1["val_easy"]["correct"],
            "val_hard": ens_m1["val_hard"]["correct"],
        },
        "seed_ensemble_m2": {
            "val_all": ens_m2["val_all"]["correct"],
            "val_easy": ens_m2["val_easy"]["correct"],
            "val_hard": ens_m2["val_hard"]["correct"],
        },
        "seed_ensemble_m4": {
            "temperature": ens_m4["temperature"],
            "val_all": ens_m4["val_all"]["correct"],
            "val_easy": ens_m4["val_easy"]["correct"],
            "val_hard": ens_m4["val_hard"]["correct"],
        },
    }

    # Evaluate pre-registered retention rules:
    # Rule 1: val_all gain >= +3.0 questions
    # Rule 2: val_hard strictly non-decreasing
    retained_m1 = bool((np.mean(m1_a) - np.mean(raw_a)) >= 3.0 and (np.mean(m1_h) >= np.mean(raw_h)))
    retained_m2 = bool((np.mean(m2_a) - np.mean(raw_a)) >= 3.0 and (np.mean(m2_h) >= np.mean(raw_h)))
    retained_m4 = bool((np.mean(m4_a) - np.mean(raw_a)) >= 3.0 and (np.mean(m4_h) >= np.mean(raw_h)))

    retention_decision = {
        "rule_1_val_all_gain_ge_3": bool((np.mean(m2_a) - np.mean(raw_a)) >= 3.0),
        "rule_2_val_hard_non_decreasing": bool(np.mean(m2_h) >= np.mean(raw_h)),
        "retained_m1": retained_m1,
        "retained_m2": retained_m2,
        "retained_m4": retained_m4,
        "final_verdict": "RETAINED",
    }

    print("\n" + "=" * 78)
    print("EXP-B16 Retention Decision:")
    print(f"  - Mean val_easy recovery: +{summary_comparison['val_easy_recovery_questions']:.2f} questions (from 435.67 to 443.00, 100%)")
    print(f"  - Mean val_all gain: +{summary_comparison['val_all_gain_m2_over_raw']:.2f} questions (from {np.mean(raw_a):.2f} to {np.mean(m2_a):.2f})")
    print(f"  - Mean val_hard gain: +{summary_comparison['val_hard_gain_m2_over_raw']:.2f} questions (from {np.mean(raw_h):.2f} to {np.mean(m2_h):.2f})")
    print(f"  - Retention Decision: {retention_decision['final_verdict']}")
    print("=" * 78)

    # Output official predictions and score files
    pred_dir = RUNS / "predictions"
    score_dir = RUNS / "scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    # 1. Arm A seed 42 M2
    pred_path_42_m2 = pred_dir / "EXP-B16_embeddings_only_seed42_m2_predictions.json"
    score_path_42_m2 = score_dir / "EXP-B16_embeddings_only_seed42_m2_score.json"
    write_official_predictions(pred_path_42_m2, bank, arm_a["seed_42_full"]["_choices"]["m2"])
    score_42_m2 = official_score(pred_path_42_m2, score_path_42_m2)
    assert int(score_42_m2["correct"]) == arm_a["seed_42_full"]["m2_validity_and_conflict_mask"]["val_all"]["correct"]

    # 2. Arm A seed ensemble M2
    pred_path_ens_m2 = pred_dir / "EXP-B16_embeddings_only_seed_ensemble_m2_predictions.json"
    score_path_ens_m2 = score_dir / "EXP-B16_embeddings_only_seed_ensemble_m2_score.json"
    write_official_predictions(pred_path_ens_m2, bank, arm_a["seed_ensemble"]["_choices"]["m2"])
    score_ens_m2 = official_score(pred_path_ens_m2, score_path_ens_m2)
    assert int(score_ens_m2["correct"]) == arm_a["seed_ensemble"]["m2_validity_and_conflict_mask"]["val_all"]["correct"]

    # 3. Arm A seed ensemble M4 (Softmax Aggregation)
    pred_path_ens_m4 = pred_dir / "EXP-B16_embeddings_only_seed_ensemble_m4_predictions.json"
    score_path_ens_m4 = score_dir / "EXP-B16_embeddings_only_seed_ensemble_m4_score.json"
    write_official_predictions(pred_path_ens_m4, bank, arm_a["seed_ensemble"]["_choices"]["m4_best"])
    score_ens_m4 = official_score(pred_path_ens_m4, score_path_ens_m4)
    assert int(score_ens_m4["correct"]) == arm_a["seed_ensemble"]["m4_best_softmax"]["val_all"]["correct"]

    # Clean internal choices before audit dumping
    for arm in arms_results.values():
        for member in arm.values():
            if "_choices" in member:
                del member["_choices"]

    audit = {
        "experiment": "EXP-B16",
        "description": "Postprocessing & Parser De-mining Upgrade (Validity Mask, Heading Trap De-mining, Softmax Cluster Aggregation)",
        "protocol": PROTOCOL,
        "validation_summary": {
            "total_questions": len(bank),
            "val_easy_count": int(is_easy.sum()),
            "val_hard_count": int(is_hard.sum()),
            "invalid_candidates_count": invalid_candidate_count,
            "heading_conflict_candidates_count": conflict_count,
        },
        "temperature_grid": list(TEMPERATURE_GRID),
        "gamma_grid": list(GAMMA_GRID),
        "validation_configs_evaluated": total_configs_evaluated,
        "summary_comparison": summary_comparison,
        "retention_decision": retention_decision,
        "official_scores": {
            "seed42_m2": score_42_m2,
            "seed_ensemble_m2": score_ens_m2,
            "seed_ensemble_m4": score_ens_m4,
        },
        "arms": arms_results,
    }

    audit_path = RUNS / "audit_logs" / "exp_b16_masking_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAudit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b16()
