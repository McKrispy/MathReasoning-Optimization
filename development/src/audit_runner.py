"""INFRA-03: External Resource, Latency, and Asset Size Auditor.

Executes predict.py and score_predictions.py under strict 4-thread CPU constraints,
measures external wall-clock latency, scans uncompressed asset size in model_bundle,
computes val_all and val_hard metrics, and emits structured audit logs to dev/runs/audit_logs/.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Optional

# Path configuration
DEV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEV_ROOT.parent
OFFICIAL_CODE_DIR = PROJECT_ROOT / 'DASE7506-MP2' / 'code'

if str(DEV_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(DEV_ROOT / 'src'))
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from common_utils import load_dataset, load_gold_references, build_label_matrix, get_hard_subset_mask

MAX_ASSET_BYTES = 16 * 1024 * 1024  # 16 MiB
MAX_WALL_SECONDS = 30.0             # 30.0 seconds


def scan_asset_size_bytes(directory: Path) -> int:
    """Recursively calculate uncompressed size of all files in directory."""
    if not directory.exists():
        return 0
    total = 0
    for p in directory.rglob('*'):
        if p.is_file() and p.name != '.gitkeep':
            total += p.stat().st_size
    return total


def run_benchmark(
    exp_id: str = "EXP-B01",
    split: str = "validation",
    method_file: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    output_log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    method = method_file or (DEV_ROOT / 'src' / 'submission.py')
    checkpoint = checkpoint_dir or (DEV_ROOT / 'runs' / 'model_bundle')
    log_dir = output_log_dir or (DEV_ROOT / 'runs' / 'audit_logs')
    pred_dir = DEV_ROOT / 'runs' / 'predictions'
    score_dir = DEV_ROOT / 'runs' / 'scores'

    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    pred_out = pred_dir / f"{exp_id}_{split}_predictions.json"
    score_out = score_dir / f"{exp_id}_{split}_score.json"

    # 1. Asset size audit
    asset_bytes = scan_asset_size_bytes(checkpoint)
    asset_mib = asset_bytes / (1024 * 1024)
    asset_compliant = asset_bytes <= MAX_ASSET_BYTES

    # 2. Environment with 4-thread CPU constraints
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    # 3. Benchmark Prediction (External Wall-Clock)
    cmd_predict = [
        sys.executable,
        str(OFFICIAL_CODE_DIR / 'predict.py'),
        '--method', str(method),
        '--split', split,
        '--output', str(pred_out),
    ]
    if checkpoint.exists() and any(p for p in checkpoint.iterdir() if p.name != '.gitkeep'):
        cmd_predict.extend(['--checkpoint', str(checkpoint)])

    t0 = time.perf_counter()
    pred_res = subprocess.run(cmd_predict, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    t1 = time.perf_counter()
    predict_wall_time = t1 - t0

    if pred_res.returncode != 0:
        raise RuntimeError(f"predict.py failed (code {pred_res.returncode}):\n{pred_res.stderr}")

    # 4. Benchmark Scoring (External Wall-Clock)
    cmd_score = [
        sys.executable,
        str(OFFICIAL_CODE_DIR / 'score_predictions.py'),
        '--split', split,
        '--predictions', str(pred_out),
        '--output', str(score_out),
    ]
    t2 = time.perf_counter()
    score_res = subprocess.run(cmd_score, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    t3 = time.perf_counter()
    score_wall_time = t3 - t2

    if score_res.returncode != 0:
        raise RuntimeError(f"score_predictions.py failed (code {score_res.returncode}):\n{score_res.stderr}")

    total_wall_time = predict_wall_time + score_wall_time
    latency_compliant = total_wall_time <= MAX_WALL_SECONDS

    # 5. Extract scores and calculate Hard subset accuracy
    official_score_data = json.loads(score_out.read_text())
    acc_all = official_score_data['accuracy']
    correct_all = official_score_data['correct']
    n_all = official_score_data['n']

    pred_data = json.loads(pred_out.read_text())
    choices = pred_data['selected']

    # Hard subset computation
    bank, _ = load_dataset(split)
    gold = load_gold_references(split)
    y = build_label_matrix(bank, gold)
    hard_mask = get_hard_subset_mask(bank)

    n_hard = int(hard_mask.sum())
    correct_hard = sum(1 for i in range(len(bank)) if hard_mask[i] and y[i, choices[i]] == 1)
    acc_hard = 100.0 * correct_hard / n_hard if n_hard > 0 else 0.0

    # 6. Build structured audit record
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    audit_record: Dict[str, Any] = {
        "timestamp": timestamp_str,
        "exp_id": exp_id,
        "split": split,
        "method": str(method),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "predictions_file": str(pred_out),
        "scores_file": str(score_out),
        "metrics": {
            "val_all_accuracy": round(acc_all, 4),
            "val_all_correct": correct_all,
            "val_all_total": n_all,
            "val_hard_accuracy": round(acc_hard, 4),
            "val_hard_correct": correct_hard,
            "val_hard_total": n_hard,
        },
        "resources": {
            "predict_wall_seconds": round(predict_wall_time, 4),
            "score_wall_seconds": round(score_wall_time, 4),
            "total_wall_seconds": round(total_wall_time, 4),
            "latency_compliant_le_30s": latency_compliant,
            "asset_uncompressed_bytes": asset_bytes,
            "asset_uncompressed_mib": round(asset_mib, 4),
            "asset_compliant_le_16mib": asset_compliant,
            "cpu_threads_limited_to": 4,
        },
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
        }
    }

    # 7. Write audit log JSON
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_log_path = log_dir / f"audit_{exp_id}_{split}_{timestamp_file}.json"
    audit_log_path.write_text(json.dumps(audit_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 8. Print formatted summary table
    print("\n" + "=" * 65)
    print(f" AUDIT BENCHMARK REPORT: {exp_id} on [{split}]")
    print("=" * 65)
    print(f" Timestamp:        {timestamp_str}")
    print(f" Accuracy (All):   {acc_all:.2f}% ({correct_all}/{n_all})")
    print(f" Accuracy (Hard):  {acc_hard:.2f}% ({correct_hard}/{n_hard})")
    print("-" * 65)
    print(f" Wall-clock Time:  {total_wall_time:.3f} s  (Predict: {predict_wall_time:.3f}s, Score: {score_wall_time:.3f}s)")
    print(f" Latency Limit:    {'PASS (<= 30.0s)' if latency_compliant else 'FAIL (> 30.0s)'}")
    print(f" Asset Size:       {asset_bytes} bytes ({asset_mib:.3f} MiB)")
    print(f" Asset Limit:      {'PASS (<= 16.0 MiB)' if asset_compliant else 'FAIL (> 16.0 MiB)'}")
    print(f" Audit Log Saved:  {audit_log_path.relative_to(PROJECT_ROOT)}")
    print("=" * 65 + "\n")

    return audit_record


def main():
    parser = argparse.ArgumentParser(description="Run audit benchmark for MP2 models.")
    parser.add_argument("--exp-id", type=str, default="EXP-B01", help="Experiment batch ID")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"], help="Data split")
    parser.add_argument("--method", type=Path, default=None, help="Path to submission.py")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model bundle")
    args = parser.parse_args()

    run_benchmark(
        exp_id=args.exp_id,
        split=args.split,
        method_file=args.method,
        checkpoint_dir=args.checkpoint,
    )


if __name__ == "__main__":
    main()
