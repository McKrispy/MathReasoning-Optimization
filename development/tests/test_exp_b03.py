"""Test for EXP-B03 reproducibility and contract integrity."""
from pathlib import Path
import sys
import json
import numpy as np

# Ensure dev/src is in sys.path
DEV_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b03_logreg import run_exp_b03


def test_exp_b03_integrity():
    results = run_exp_b03()
    
    # 1. Contingency tables
    assert len(results["contingency_tables"]) == 4
    for entry in results["contingency_tables"]:
        assert entry["action"] == "CONSISTENT"
        assert not entry["opposite_direction"]
    assert len(results["cols_to_delete"]) == 0

    # 2. Fit 1 (Consensus Core)
    fit1 = results["fit1_consensus_core"]
    assert fit1["num_features"] == 4
    assert set(fit1["features"]) == {'is_valid', 'cluster_ratio', 'is_unique_majority', 'is_tied_majority'}
    # Accuracy strictly matches baseline
    assert abs(fit1["acc_val_hard"] - 0.690141) < 1e-4
    assert abs(fit1["acc_val_all"] - 0.862155) < 1e-4

    # 3. Fit 2 (Consensus Core + other eligible families)
    fit2 = results["fit2_full_retained"]
    assert fit2["num_features"] == 32
    assert fit2["acc_val_hard"] > fit1["acc_val_hard"]
    assert fit2["acc_val_all"] >= fit1["acc_val_all"]

    # 4. Check audit log existence
    audit_file = DEV_SRC.parent / 'runs' / 'audit_logs' / 'exp_b03_logreg_audit.json'
    assert audit_file.exists()
