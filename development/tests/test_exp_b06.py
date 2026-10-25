"""Test for EXP-B06 pointwise verifier on raw embeddings reproducibility."""
from pathlib import Path
import sys
import json
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b06_pointwise import run_exp_b06


def test_exp_b06_integrity():
    results = run_exp_b06()

    # 1. Performance checks
    acc_vh = results["accuracy"]["val_hard"]["acc"]
    acc_va = results["accuracy"]["val_all"]["acc"]
    assert acc_vh > 0.70, f"Expected strong pointwise val_hard accuracy, got {acc_vh}"
    assert acc_va > 0.85, f"Expected strong pointwise val_all accuracy, got {acc_va}"

    # 2. Asset size and latency
    assert results["asset_size_mib"] < 1.0, f"Expected compact model, got {results['asset_size_mib']} MiB"
    assert results["latency_seconds_val_all"] < 1.0

    # 3. Model sweeps
    assert len(results["lr_sweep"]) == 4
    assert "mlp_results" in results

    # 4. Check audit logs and output files
    runs_dir = DEV_SRC.parent / 'runs'
    assert (runs_dir / 'audit_logs' / 'exp_b06_pointwise_audit.json').exists()
    assert (runs_dir / 'predictions' / 'EXP-B06_pointwise_predictions.json').exists()
    assert (runs_dir / 'scores' / 'EXP-B06_pointwise_score.json').exists()
