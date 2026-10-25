"""EXP-B20: B17 Reproducible Replay, Resource Audit, and Strict OOF Baseline.

This script executes:
1. Asset packaging and expanded uncompressed byte-level inventory in dev/runs/exp_b20/model_assets/.
2. In-process replay verification against historical B17 predictions (300/355, 743/798).
3. Subprocess external wall-clock latency measurement with 4-thread CPU cap.
4. Strict fold-contained Out-Of-Fold (OOF) reconstruction:
   - Fold-local SVD fitting (no pre-splitting data leakage)
   - Raw structure features on Tree R5 (matching full model scale)
   - 3-seed fold model averaging for attention views (matching full view ensemble)
5. Audit log generation:
   - dev/runs/audit_logs/exp_b20_resource_audit.json
   - dev/runs/audit_logs/exp_b20_oof_audit.json
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

DEV_SRC = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
OFFICIAL_CODE = WORKSPACE / "DASE7506-MP2" / "code"

if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

import common  # type: ignore
from common_utils import get_hard_subset_mask, load_dataset, load_gold_references
from experiments.exp_b07_cross_attn import (
    HPARAMS,
    kfold_indices,
    standardize_apply,
    standardize_fit,
)
from experiments.exp_b15_tree import HGB_PARAMS
from experiments.multiseed import (
    FOLD_PARTITION_SEED,
    RUNS,
    SEEDS,
    ArmInput,
    SplitData,
    environment_record,
    reasoning_struct,
)
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
MAX_BUNDLE_MIB = 16.0
MAX_WALLCLOCK_SECONDS = 30.0

EXP_B20_DIR = RUNS / "exp_b20"
ASSETS_DIR = EXP_B20_DIR / "model_assets"
AUDIT_DIR = RUNS / "audit_logs"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_uncompressed_bytes(path: Path) -> Tuple[int, int]:
    """Returns (file_bytes, uncompressed_expanded_bytes)."""
    file_bytes = path.stat().st_size
    if path.suffix == ".npz":
        # Sum of nbytes of uncompressed arrays + minimal NPY format header estimation (64 bytes each)
        with np.load(path) as data:
            expanded = 0
            for k in data.files:
                arr = data[k]
                expanded += arr.nbytes + 64
        return file_bytes, expanded
    else:
        # pkl, json, txt: raw uncompressed file size
        return file_bytes, file_bytes


def step1_prepare_assets() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("Step 1: Packaging B17 Inference Assets into dev/runs/exp_b20/model_assets/")
    print("=" * 78)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Arm A (embeddings_only) 3 seeds
    for s in SEEDS:
        src = RUNS / f"exp_b12/seed_{s}/arm_embeddings_only_full.npz"
        dst = ASSETS_DIR / f"arm_a_seed_{s}.npz"
        shutil.copy2(src, dst)

    # 2. Arm B (embeddings_plus_g01_g03) 3 seeds
    for s in SEEDS:
        src = RUNS / f"exp_b12/seed_{s}/arm_embeddings_plus_g01_g03_full.npz"
        dst = ASSETS_DIR / f"arm_b_seed_{s}.npz"
        shutil.copy2(src, dst)

    # 3. Tree R5 model
    src_tree = RUNS / "exp_b15/seed_42/M04a_train_all_plus_svd.pkl"
    dst_tree = ASSETS_DIR / "tree_r5.pkl"
    shutil.copy2(src_tree, dst_tree)

    # 4. SVD components
    src_svd = RUNS / "exp_b15/feat_g04b_svd.npz"
    dst_svd = ASSETS_DIR / "svd_components.npz"
    shutil.copy2(src_svd, dst_svd)

    # 5. Configuration
    config = {
        "model_id": "MOD-M03+MOD-M04a+POST-P06",
        "description": "EXP-B17 3-View Soft Stacking Inference Bundle",
        "protocol": PROTOCOL,
        "weights": [0.6519943983184848, 0.2773940600630942, 0.07061154161842091],
        "temperature": 0.5,
        "mask_penalty": 1000000000.0,
        "seeds": [42, 43, 44],
        "views": ["arm_a_ensemble", "arm_b_ensemble", "tree_r5"],
        "features": {
            "consensus_core": ["is_valid", "cluster_ratio", "is_unique_majority", "is_tied_majority"],
            "symbolic": [
                "num_equations",
                "num_correct_equations",
                "num_wrong_equations",
                "has_arithmetic_error",
                "candidate_number_count",
                "question_condition_coverage",
                "missing_condition_count",
                "has_missing_condition",
                "wrong_eq_diff_mean",
                "wrong_eq_rank",
                "condition_coverage_diff_mean",
                "condition_coverage_rank",
            ],
            "svd_dimensions": 64,
            "svd_total_dimensions": 128,
        },
    }
    config_path = ASSETS_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Inventory calculation
    inventory = []
    total_file_bytes = 0
    total_uncompressed_bytes = 0

    purpose_map = {
        "arm_a_seed_42.npz": "View 1 (Arm A embeddings-only cross-attention weights, seed 42)",
        "arm_a_seed_43.npz": "View 1 (Arm A embeddings-only cross-attention weights, seed 43)",
        "arm_a_seed_44.npz": "View 1 (Arm A embeddings-only cross-attention weights, seed 44)",
        "arm_b_seed_42.npz": "View 2 (Arm B symbolic cross-attention weights & normalization, seed 42)",
        "arm_b_seed_43.npz": "View 2 (Arm B symbolic cross-attention weights & normalization, seed 43)",
        "arm_b_seed_44.npz": "View 2 (Arm B symbolic cross-attention weights & normalization, seed 44)",
        "tree_r5.pkl": "View 3 (Tree R5 HistGradientBoostingClassifier model)",
        "svd_components.npz": "TruncatedSVD 64-dim projection matrix",
        "config.json": "Inference configuration, column schema, stacking weights, and post-processing params",
    }

    for p in sorted(ASSETS_DIR.glob("*")):
        if p.is_file():
            f_bytes, uncompressed = compute_uncompressed_bytes(p)
            digest = sha256_of_file(p)
            total_file_bytes += f_bytes
            total_uncompressed_bytes += uncompressed
            item = {
                "file": p.name,
                "purpose": purpose_map.get(p.name, "Asset"),
                "file_bytes": f_bytes,
                "uncompressed_bytes": uncompressed,
                "sha256": digest,
            }
            inventory.append(item)
            print(f"  {p.name:25s} | File: {f_bytes:>8,d} B | Uncompressed: {uncompressed:>8,d} B | SHA256: {digest[:12]}...")

    total_uncompressed_mib = total_uncompressed_bytes / (1024 * 1024)
    print(f"\nTotal asset files: {len(inventory)}")
    print(f"Total file bytes:         {total_file_bytes:,d} B ({total_file_bytes / (1024*1024):.4f} MiB)")
    print(f"Total uncompressed bytes: {total_uncompressed_bytes:,d} B ({total_uncompressed_mib:.4f} MiB)")
    print(f"Compliance check (<= 16.0 MiB): {'PASS' if total_uncompressed_mib <= MAX_BUNDLE_MIB else 'FAIL'}")

    return {
        "inventory": inventory,
        "total_file_bytes": total_file_bytes,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "total_uncompressed_mib": total_uncompressed_mib,
        "limit_mib": MAX_BUNDLE_MIB,
        "compliant": bool(total_uncompressed_mib <= MAX_BUNDLE_MIB),
    }


def step2_in_process_replay() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("Step 2: In-Process Replay Verification of candidate_submission.py")
    print("=" * 78)

    if str(EXP_B20_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_B20_DIR))
    from candidate_submission import predict as candidate_predict

    val_bank, val_arrays = common.load_inputs("validation", OFFICIAL_CODE / "data")
    gold_val = load_gold_references("validation")
    gold_answers = [common.parse_answer(q["answer"]) for q in gold_val]
    is_hard = np.array(get_hard_subset_mask(val_bank), dtype=bool)

    t0 = time.perf_counter()
    choices = candidate_predict(val_bank, val_arrays, ASSETS_DIR)
    in_process_seconds = time.perf_counter() - t0

    # Compare with historical B17 predictions
    hist_b17_path = RUNS / "predictions/EXP-B17_soft_stacking_3view_predictions.json"
    hist_b17_pred = json.loads(hist_b17_path.read_text(encoding="utf-8"))["selected"]

    match_count = sum(c1 == c2 for c1, c2 in zip(choices, hist_b17_pred))
    match_pct = 100.0 * match_count / len(choices)

    hits = np.array(
        [common.parse_answer(val_bank[i]["candidates"][choices[i]]["text"]) == gold_answers[i] for i in range(len(val_bank))],
        dtype=bool,
    )
    is_easy = ~is_hard

    val_all_correct = int(hits.sum())
    val_hard_correct = int(hits[is_hard].sum())
    val_easy_correct = int(hits[is_easy].sum())

    print(f"Replay Execution Time: {in_process_seconds:.2f}s")
    print(f"Agreement with Historical B17 Predictions: {match_count} / {len(choices)} ({match_pct:.2f}%)")
    print(f"Score - val_all:  {val_all_correct} / {len(val_bank)} ({100.0 * val_all_correct / len(val_bank):.2f}%) [B17 Benchmark: 743]")
    print(f"Score - val_hard: {val_hard_correct} / {is_hard.sum()} ({100.0 * val_hard_correct / is_hard.sum():.2f}%) [B17 Benchmark: 300]")
    print(f"Score - val_easy: {val_easy_correct} / {is_easy.sum()} ({100.0 * val_easy_correct / is_easy.sum():.2f}%) [B17 Benchmark: 443]")

    assert val_all_correct == 743, f"val_all mismatch: expected 743, got {val_all_correct}"
    assert val_hard_correct == 300, f"val_hard mismatch: expected 300, got {val_hard_correct}"
    assert val_easy_correct == 443, f"val_easy mismatch: expected 443, got {val_easy_correct}"
    print("Exact Numerical Parity with B17 Historical Score Confirmed!")

    return {
        "match_count_vs_historical_b17": match_count,
        "match_percentage": match_pct,
        "in_process_seconds": in_process_seconds,
        "val_all": {"correct": val_all_correct, "total": len(val_bank), "accuracy": 100.0 * val_all_correct / len(val_bank)},
        "val_hard": {"correct": val_hard_correct, "total": int(is_hard.sum()), "accuracy": 100.0 * val_hard_correct / int(is_hard.sum())},
        "val_easy": {"correct": val_easy_correct, "total": int(is_easy.sum()), "accuracy": 100.0 * val_easy_correct / int(is_easy.sum())},
    }


def step3_subprocess_latency_benchmark() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("Step 3: External Process Wall-Clock Benchmark (4-thread CPU Cap)")
    print("=" * 78)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    candidate_sub = EXP_B20_DIR / "candidate_submission.py"
    pred_out = EXP_B20_DIR / "validation-predictions.json"
    score_out = EXP_B20_DIR / "validation-score.json"

    # Command 1: Fresh predict.py process
    cmd_predict = [
        sys.executable,
        str(OFFICIAL_CODE / "predict.py"),
        "--method",
        str(candidate_sub),
        "--checkpoint",
        str(ASSETS_DIR),
        "--split",
        "validation",
        "--output",
        str(pred_out),
    ]

    print(f"Running command: {' '.join(cmd_predict)}")
    t0_pred = time.perf_counter()
    proc_pred = subprocess.run(cmd_predict, capture_output=True, text=True, env=env)
    wallclock_predict = time.perf_counter() - t0_pred

    if proc_pred.returncode != 0:
        print("STDERR:", proc_pred.stderr)
        raise RuntimeError(f"predict.py failed with returncode {proc_pred.returncode}")
    print(f"STDOUT: {proc_pred.stdout.strip()}")
    print(f"External wallclock predict: {wallclock_predict:.3f}s")

    # Command 2: Independent score_predictions.py process
    cmd_score = [
        sys.executable,
        str(OFFICIAL_CODE / "score_predictions.py"),
        "--split",
        "validation",
        "--predictions",
        str(pred_out),
        "--output",
        str(score_out),
    ]

    print(f"Running command: {' '.join(cmd_score)}")
    t0_score = time.perf_counter()
    proc_score = subprocess.run(cmd_score, capture_output=True, text=True, env=env)
    wallclock_score = time.perf_counter() - t0_score

    if proc_score.returncode != 0:
        print("STDERR:", proc_score.stderr)
        raise RuntimeError(f"score_predictions.py failed with returncode {proc_score.returncode}")
    print(f"STDOUT: {proc_score.stdout.strip()}")
    print(f"External wallclock score:   {wallclock_score:.3f}s")

    total_wallclock = wallclock_predict + wallclock_score
    print(f"Total end-to-end CPU wallclock (predict + score): {total_wallclock:.3f}s")
    print(f"Compliance check (<= 30.0s): {'PASS' if total_wallclock <= MAX_WALLCLOCK_SECONDS else 'FAIL'}")

    score_data = json.loads(score_out.read_text(encoding="utf-8"))
    assert int(score_data["correct"]) == 743, f"Score mismatch: expected 743, got {score_data['correct']}"

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "threads_configured": 4,
        "wallclock_predict_seconds": wallclock_predict,
        "wallclock_score_seconds": wallclock_score,
        "wallclock_total_seconds": total_wallclock,
        "limit_seconds": MAX_WALLCLOCK_SECONDS,
        "compliant": bool(total_wallclock <= MAX_WALLCLOCK_SECONDS),
        "official_score": score_data,
    }


def step4_strict_oof_reconstruction() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("Step 4: Strict Fold-Contained OOF Baseline Reconstruction")
    print("=" * 78)

    train = SplitData("train")
    val = SplitData("validation")
    gold_val = load_gold_references("validation")
    gold_answers = [common.parse_answer(q["answer"]) for q in gold_val]

    n_train = len(train.bank)
    n_val = len(val.bank)
    is_hard = np.array(val.hard, dtype=bool)

    print(f"Dataset: train={n_train}, validation={n_val}")

    # Extract structured reasoning features (16 cols)
    struct_train, _, _ = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)

    folds = kfold_indices(n_train, 5, FOLD_PARTITION_SEED)
    train_hard_mask = np.array(train.hard, dtype=bool)

    # Prepare strict OOF arrays
    oof_a_strict = np.zeros((n_train, 4), dtype=np.float32)
    oof_b_strict = np.zeros((n_train, 4), dtype=np.float32)
    oof_tree_strict = np.zeros((n_train, 4), dtype=np.float32)

    t_strict_start = time.perf_counter()

    for k, held in enumerate(folds):
        fit_rows = np.concatenate([folds[o] for o in range(5) if o != k])

        # 1. View 1 (Arm A): 3-seed average of fold k models
        xa_held = train.emb[held]
        logits_a_seeds = []
        for s in SEEDS:
            wa = dict(np.load(RUNS / f"exp_b12/seed_{s}/arm_embeddings_only_fold{k}.npz"))
            net_a = CrossCandidateAttentionNet(wa)
            logits_a_seeds.append(net_a.forward_numpy(xa_held))
        oof_a_strict[held] = np.mean(logits_a_seeds, axis=0)

        # 2. View 2 (Arm B): 3-seed average of fold k models (each normalized with its fold mean/std)
        logits_b_seeds = []
        for s in SEEDS:
            wb = dict(np.load(RUNS / f"exp_b12/seed_{s}/arm_embeddings_plus_g01_g03_fold{k}.npz"))
            m, std = wb["struct_mean"], wb["struct_std"]
            struct_held_norm = (struct_train[held] - m) / np.maximum(std, 1e-6)
            xb_held = np.concatenate([train.emb[held], struct_held_norm], axis=-1)
            net_b = CrossCandidateAttentionNet(wb)
            logits_b_seeds.append(net_b.forward_numpy(xb_held))
        oof_b_strict[held] = np.mean(logits_b_seeds, axis=0)

        # 3. View 3 (Tree R5): fold-local SVD + raw structure features
        svd_k = TruncatedSVD(n_components=64, random_state=42)
        fit_flat_emb = train.emb[fit_rows].reshape(-1, 1024)
        svd_k.fit(fit_flat_emb)

        # Transform fit_rows and held
        fit_proj = (fit_flat_emb @ svd_k.components_.T).reshape(len(fit_rows), 4, 64)
        fit_centered = fit_proj - fit_proj.mean(axis=1, keepdims=True)
        fit_svd_128 = np.concatenate([fit_proj, fit_centered], axis=-1).astype(np.float32)

        held_flat_emb = train.emb[held].reshape(-1, 1024)
        held_proj = (held_flat_emb @ svd_k.components_.T).reshape(len(held), 4, 64)
        held_centered = held_proj - held_proj.mean(axis=1, keepdims=True)
        held_svd_128 = np.concatenate([held_proj, held_centered], axis=-1).astype(np.float32)

        # Tree features: raw struct_train + svd_128 (matching full model scale)
        x_tree_fit = np.concatenate([struct_train[fit_rows], fit_svd_128], axis=-1)
        x_tree_held = np.concatenate([struct_train[held], held_svd_128], axis=-1)

        tree_fold = HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS)
        tree_fold.fit(x_tree_fit.reshape(-1, 144), train.labels[fit_rows].reshape(-1))
        oof_tree_strict[held] = tree_fold.decision_function(x_tree_held.reshape(-1, 144)).reshape(-1, 4)

    t_strict_oof = time.perf_counter() - t_strict_start
    print(f"Strict fold-contained OOF generated in {t_strict_oof:.2f}s.")

    # Standardize OOF within problem
    s_a_strict = standardize_within_problem(oof_a_strict)
    s_b_strict = standardize_within_problem(oof_b_strict)
    s_tree_strict = standardize_within_problem(oof_tree_strict)

    # Solve optimal convex combination weights on train_hard
    w_strict = fit_convex_combination_weights(
        [s_a_strict, s_b_strict, s_tree_strict],
        train.labels,
        subset_mask=train_hard_mask,
    )

    w_b17_historical = [0.6519943983184848, 0.2773940600630942, 0.07061154161842091]

    print(f"Historical B17 Stacking Weights: {[round(float(x), 4) for x in w_b17_historical]}")
    print(f"Strict Reconstructed Weights:     {[round(float(x), 4) for x in w_strict]}")

    # Evaluate validation set with strict weights
    logits_a_dict = np.load(RUNS / "exp_b12/val_logits_embeddings_only.npz")
    logits_b_dict = np.load(RUNS / "exp_b12/val_logits_embeddings_plus_g01_g03.npz")
    tree_logits_dict = np.load(RUNS / "exp_b15/val_logits_M04a_train_all_plus_svd.npz")

    v1_val = logits_a_dict["seed_ensemble"]
    v2_val = logits_b_dict["seed_ensemble"]
    v4_val = tree_logits_dict["seed_42"]

    s1_val = standardize_within_problem(v1_val)
    s2_val = standardize_within_problem(v2_val)
    s4_val = standardize_within_problem(v4_val)

    stacked_strict = w_strict[0] * s1_val + w_strict[1] * s2_val + w_strict[2] * s4_val
    z_strict = apply_validity_mask(stacked_strict, val.bank, penalty=1e9)
    z_strict = apply_conflict_penalty(z_strict, val.bank, penalty=1e9)
    c_strict = SoftmaxClusterAggregator(temperature=0.5).aggregate(val.bank, z_strict)

    hits_strict = np.array(
        [common.parse_answer(val.bank[i]["candidates"][c_strict[i]]["text"]) == gold_answers[i] for i in range(len(val.bank))],
        dtype=bool,
    )
    val_all_strict = int(hits_strict.sum())
    val_hard_strict = int(hits_strict[is_hard].sum())

    print(f"Validation Performance with Strict OOF Weights:")
    print(f"  val_all:  {val_all_strict} / 798 ({100.0 * val_all_strict / 798:.2f}%) [Historical: 743]")
    print(f"  val_hard: {val_hard_strict} / 355 ({100.0 * val_hard_strict / 355:.2f}%) [Historical: 300]")

    return {
        "generation_time_seconds": t_strict_oof,
        "historical_b17_weights": [float(x) for x in w_b17_historical],
        "strict_oof_weights": [float(x) for x in w_strict],
        "validation_with_strict_weights": {
            "val_all_correct": val_all_strict,
            "val_all_accuracy": 100.0 * val_all_strict / 798,
            "val_hard_correct": val_hard_strict,
            "val_hard_accuracy": 100.0 * val_hard_strict / 355,
        },
        "validation_with_historical_weights": {
            "val_all_correct": 743,
            "val_all_accuracy": 100.0 * 743 / 798,
            "val_hard_correct": 300,
            "val_hard_accuracy": 100.0 * 300 / 355,
        },
        "oof_matrix_metadata": {
            "train_questions": n_train,
            "folds": 5,
            "arm_a_seeds": list(SEEDS),
            "arm_b_seeds": list(SEEDS),
            "tree_feature_dim": 144,
            "fold_local_svd_fit": True,
            "raw_struct_feature_scale": True,
        },
    }


def main():
    print("=" * 78)
    print("Executing EXP-B20: B17 Replay, Resource Audit, and Strict OOF Baseline")
    print("=" * 78)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_B20_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Asset packaging & inventory
    asset_res = step1_prepare_assets()

    # 2. In-process replay verification
    replay_res = step2_in_process_replay()

    # 3. External subprocess wall-clock benchmark
    benchmark_res = step3_subprocess_latency_benchmark()

    # 4. Strict OOF baseline reconstruction
    oof_res = step4_strict_oof_reconstruction()

    # 5. Write audit logs
    env_rec = environment_record(torch.device("cpu"))

    resource_audit = {
        "experiment": "EXP-B20",
        "description": "B17 Independent Candidate Replay & Resource Audit",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env_rec,
        "candidate_entrypoint": "dev/runs/exp_b20/candidate_submission.py",
        "checkpoint_dir": "dev/runs/exp_b20/model_assets",
        "assets": asset_res,
        "in_process_replay": replay_res,
        "external_benchmark_4_threads": benchmark_res,
        "verdict": {
            "asset_size_compliant": asset_res["compliant"],
            "wallclock_compliant": benchmark_res["compliant"],
            "exact_score_parity": bool(replay_res["val_all"]["correct"] == 743 and replay_res["val_hard"]["correct"] == 300),
            "status": "PASSED" if (asset_res["compliant"] and benchmark_res["compliant"]) else "FAILED",
        },
    }
    resource_audit_path = AUDIT_DIR / "exp_b20_resource_audit.json"
    resource_audit_path.write_text(json.dumps(resource_audit, indent=2) + "\n", encoding="utf-8")
    print(f"\nResource audit written to: {resource_audit_path}")

    oof_audit = {
        "experiment": "EXP-B20",
        "description": "B17 Strict Fold-Contained OOF Baseline Reconstruction",
        "protocol": PROTOCOL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env_rec,
        "strict_oof": oof_res,
        "conclusion": (
            "Strict fold-contained OOF aligns preprocessing scales (raw struct features on tree) "
            "and averages fold models across 3 seeds to mirror final inference. "
            "Weights and scores provide an unbiased baseline for EXP-B21 and EXP-B22."
        ),
    }
    oof_audit_path = AUDIT_DIR / "exp_b20_oof_audit.json"
    oof_audit_path.write_text(json.dumps(oof_audit, indent=2) + "\n", encoding="utf-8")
    print(f"OOF audit written to:      {oof_audit_path}")

    print("\n" + "=" * 78)
    print("EXP-B20 Execution Complete!")
    print(f"Status: {resource_audit['verdict']['status']}")
    print(f"Uncompressed Assets: {asset_res['total_uncompressed_mib']:.3f} MiB / 16.0 MiB")
    print(f"Wall-Clock Latency:  {benchmark_res['wallclock_total_seconds']:.3f}s / 30.0s")
    print(f"Replayed Scores:     val_all={replay_res['val_all']['correct']}/798, val_hard={replay_res['val_hard']['correct']}/355")
    print("=" * 78)


if __name__ == "__main__":
    main()
