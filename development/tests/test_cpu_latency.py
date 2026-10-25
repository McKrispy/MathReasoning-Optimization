"""TEST-T03: CPU latency test under 4 threads (strict <= 30.0s limit)."""
import os
from pathlib import Path
import subprocess
import sys
import time

DEV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEV_ROOT.parent
OFFICIAL_CODE_DIR = PROJECT_ROOT / 'DASE7506-MP2' / 'code'


def run_latency_benchmark(split: str = 'validation', max_allowed_seconds: float = 30.0):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    pred_out = DEV_ROOT / 'runs' / 'predictions' / f'latency_test_{split}.json'
    score_out = DEV_ROOT / 'runs' / 'scores' / f'latency_test_{split}.json'
    checkpoint = DEV_ROOT / 'runs' / 'model_bundle'

    cmd_predict = [
        sys.executable,
        str(OFFICIAL_CODE_DIR / 'predict.py'),
        '--method', str(DEV_ROOT / 'src' / 'submission.py'),
        '--split', split,
        '--output', str(pred_out),
    ]
    if checkpoint.exists() and any(checkpoint.iterdir()):
        cmd_predict.extend(['--checkpoint', str(checkpoint)])

    cmd_score = [
        sys.executable,
        str(OFFICIAL_CODE_DIR / 'score_predictions.py'),
        '--split', split,
        '--predictions', str(pred_out),
        '--output', str(score_out),
    ]

    start = time.perf_counter()
    subprocess.run(cmd_predict, check=True, env=env, cwd=str(PROJECT_ROOT))
    subprocess.run(cmd_score, check=True, env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start

    print(f"Elapsed wall-clock time for {split} prediction + scoring: {elapsed:.2f} s")
    assert elapsed <= max_allowed_seconds, f"Latency {elapsed:.2f}s exceeded limit of {max_allowed_seconds}s"
    return elapsed


if __name__ == "__main__":
    run_latency_benchmark('validation')
    print("TEST-T03 PASSED: CPU latency is well under 30.0s.")
