"""Unit tests and automated verification for EXP-B21."""
import json
from pathlib import Path
import pytest

RUNS = Path(__file__).resolve().parents[1] / "runs"
AUDIT_JSON = RUNS / "audit_logs" / "exp_b21_g05_view_audit.json"


def test_exp_b21_audit_exists():
    assert AUDIT_JSON.exists(), f"Audit file not found: {AUDIT_JSON}"


def test_exp_b21_rejection_decision():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    gate = audit["retention_gate"]
    assert gate["verdict"] == "REJECTED"
    # Linear G05 received zero or negligible weight in OOF stacking
    w_g05 = audit["training_oof_optimization"]["historical_oof_weights"]["linear_g05"]
    assert w_g05 < 0.05, f"Expected linear G05 weight < 0.05, got {w_g05}"


def test_exp_b21_standalone_numbers():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    standalone = audit["mod_m07a_standalone"]["argmax"]
    assert standalone["val_all"]["correct"] == 705
    assert standalone["val_hard"]["correct"] == 262
