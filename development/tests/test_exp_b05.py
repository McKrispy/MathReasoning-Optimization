"""Test for EXP-B05 surface features alone and false positive analysis reproducibility."""
from pathlib import Path
import sys
import json
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b05_surface import run_exp_b05


def test_exp_b05_integrity():
    results = run_exp_b05()

    # 1. Performance checks
    acc_vh = results["accuracy"]["val_hard"]["acc"]
    acc_va = results["accuracy"]["val_all"]["acc"]
    assert acc_vh < 0.65, f"Expected surface-only val_hard accuracy to collapse, got {acc_vh}"
    assert acc_va < 0.85, f"Expected surface-only val_all accuracy to be degraded, got {acc_va}"

    # 2. Asset size and latency
    assert results["asset_size_mib"] < 1.0, f"Expected compact GBDT model, got {results['asset_size_mib']} MiB"
    assert results["latency_seconds_val_all"] < 1.0

    # 3. False Positive Analysis checks
    fp_analysis = results["false_positive_analysis"]
    assert fp_analysis["false_positives_count"] > 100
    assert fp_analysis["false_positive_rate"] >= 0.35  # At least 35% error rate on val_hard

    # Check length bias (false positives longer than true positives)
    assert fp_analysis["surface_comparison"]["false_positive_char_len"] > fp_analysis["surface_comparison"]["true_positive_char_len"]

    # Check diagnostic flaws
    assert fp_analysis["symbolic_verification_of_false_positives"]["has_arithmetic_error_count"] > 20
    assert len(fp_analysis["representative_case_studies"]) >= 3

    # 4. Check audit logs and output files
    runs_dir = DEV_SRC.parent / 'runs'
    assert (runs_dir / 'audit_logs' / 'exp_b05_surface_audit.json').exists()
    assert (runs_dir / 'predictions' / 'EXP-B05_surface_predictions.json').exists()
    assert (runs_dir / 'scores' / 'EXP-B05_surface_score.json').exists()
