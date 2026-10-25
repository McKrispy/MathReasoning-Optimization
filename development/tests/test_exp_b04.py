"""Test for EXP-B04 incremental ablation reproducibility and contract integrity."""
from pathlib import Path
import sys
import json
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b04_ablation import run_exp_b04_ablation


def test_exp_b04_integrity():
    results = run_exp_b04_ablation()

    # 1. Baseline verification
    res_base = results["baseline"]
    assert res_base["num_features"] == 4
    assert abs(res_base["val_hard"]["acc"] - 0.690141) < 1e-4

    # 2. Family ablation verification
    runs = results["runs"]
    # FEAT-G02: gain < 1.0%
    delta_g02 = runs["FEAT-G02"]["val_hard"]["acc"] - res_base["val_hard"]["acc"]
    assert delta_g02 < 0.0100

    # FEAT-G03: gain >= 1.0%
    delta_g03 = runs["FEAT-G03"]["val_hard"]["acc"] - res_base["val_hard"]["acc"]
    assert delta_g03 >= 0.0100
    assert "FEAT-G03 (Symbolic)" in results["qualified_families"]

    # FEAT-G04: gain < 1.0%
    delta_g04 = runs["FEAT-G04"]["val_hard"]["acc"] - res_base["val_hard"]["acc"]
    assert delta_g04 < 0.0100

    # 3. Check audit log existence
    audit_file = DEV_SRC.parent / 'runs' / 'audit_logs' / 'exp_b04_ablation_audit.json'
    assert audit_file.exists()
