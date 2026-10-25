"""Reproduce a supplied split with the public entrypoint and official scorer.

This controller uses standard-library modules only. Prediction and scoring
each run in a fresh process. Stored predictions are compared only afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE", "OMP_MAX_ACTIVE_LEVELS": "1",
    "CUDA_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "-1",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], environment: dict, output: Path, label: str):
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True)
    elapsed = time.perf_counter_ns() - started
    (output / f"{label}-stdout.log").write_bytes(result.stdout)
    (output / f"{label}-stderr.log").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}: " + result.stderr.decode("utf-8", errors="replace"))
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Local output directory, inside or outside the repository")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "predictions.json"
    score_path = output / "score.json"
    if prediction.exists() or score_path.exists():
        raise FileExistsError("Output already contains predictions or a score; choose a new directory")
    manifest = json.loads((ROOT / "frozen/manifest.json").read_text(encoding="utf-8"))
    for row in manifest["runtime_files"] + manifest["assets"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("Release hash mismatch: " + row["path"])

    environment = os.environ.copy()
    environment.update(ENVIRONMENT)
    for key in list(environment):
        if key.startswith("B29_") or key == "PYTHONPATH":
            environment.pop(key, None)
    cache = output / "cache"
    temporary = output / "tmp"
    cache.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR"):
        environment[key] = str(temporary)
    for key in ("XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "HF_HOME"):
        environment[key] = str(cache)

    command = [sys.executable, "-B", "code/predict.py", "--method", "code/submission.py",
               "--checkpoint", "frozen/model_bundle", "--split", args.split,
               "--output", str(prediction)]
    scorer = [sys.executable, "-B", "code/score_predictions.py", "--split", args.split,
              "--predictions", str(prediction), "--output", str(score_path)]
    start = time.perf_counter_ns()
    prediction_ns = run(command, environment, output, "prediction")
    scoring_ns = run(scorer, environment, output, "scoring")
    continuous_ns = time.perf_counter_ns() - start
    actual = json.loads(prediction.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    expected_path = ROOT / "frozen/results" / ("primary/test-predictions.json" if args.split == "test" else "validation/validation-predictions.json")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    matches = all(actual[k] == expected[k] for k in ("protocol", "split", "ids", "selected"))
    expected_count = 374 if args.split == "test" else 743
    if not matches or score["correct"] != expected_count:
        raise RuntimeError("Published frozen result did not reproduce exactly")
    summary = {
        "split": args.split, "correct": score["correct"], "n": score["n"],
        "accuracy": score["accuracy"], "exact_published_choices": matches,
        "predict_wall_seconds": prediction_ns / 1e9,
        "score_wall_seconds": scoring_ns / 1e9,
        "sum_process_wall_seconds": (prediction_ns + scoring_ns) / 1e9,
        "continuous_pair_wall_seconds": continuous_ns / 1e9,
        "cpu_numeric_threads": 1, "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(),
        "checkpoint_decision_bytes": manifest["asset_decision_bytes"],
        "no_retraining": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
