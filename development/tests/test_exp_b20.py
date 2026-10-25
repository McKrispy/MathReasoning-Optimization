"""Unit tests and automated compliance verification for EXP-B20."""
import json
from pathlib import Path
import pytest

RUNS = Path(__file__).resolve().parents[1] / "runs"
ASSETS_DIR = RUNS / "exp_b20" / "model_assets"
RESOURCE_AUDIT = RUNS / "audit_logs" / "exp_b20_resource_audit.json"
OOF_AUDIT = RUNS / "audit_logs" / "exp_b20_oof_audit.json"


def test_exp_b20_audits_exist():
    assert RESOURCE_AUDIT.exists(), f"Resource audit missing: {RESOURCE_AUDIT}"
    assert OOF_AUDIT.exists(), f"OOF audit missing: {OOF_AUDIT}"


def test_exp_b20_asset_size_compliance():
    audit = json.loads(RESOURCE_AUDIT.read_text(encoding="utf-8"))
    uncompressed_mib = audit["assets"]["total_uncompressed_mib"]
    limit_mib = audit["assets"]["limit_mib"]
    assert audit["verdict"]["asset_size_compliant"] is True
    assert uncompressed_mib <= limit_mib, f"Asset size {uncompressed_mib:.3f} MiB > {limit_mib} MiB"


def test_exp_b20_wallclock_compliance():
    audit = json.loads(RESOURCE_AUDIT.read_text(encoding="utf-8"))
    wallclock = audit["external_benchmark_4_threads"]["wallclock_total_seconds"]
    limit_seconds = audit["external_benchmark_4_threads"]["limit_seconds"]
    assert audit["verdict"]["wallclock_compliant"] is True
    assert wallclock <= limit_seconds, f"Wall-clock {wallclock:.3f}s > {limit_seconds}s"


def test_exp_b20_exact_numerical_parity():
    audit = json.loads(RESOURCE_AUDIT.read_text(encoding="utf-8"))
    replay = audit["in_process_replay"]
    assert replay["val_all"]["correct"] == 743, f"val_all mismatch: {replay['val_all']['correct']}"
    assert replay["val_hard"]["correct"] == 300, f"val_hard mismatch: {replay['val_hard']['correct']}"
    assert replay["val_easy"]["correct"] == 443, f"val_easy mismatch: {replay['val_easy']['correct']}"
    assert replay["match_count_vs_historical_b17"] == 798, f"Disagreed on {798 - replay['match_count_vs_historical_b17']} items"


def test_exp_b20_strict_oof_reconstruction():
    audit = json.loads(OOF_AUDIT.read_text(encoding="utf-8"))
    oof = audit["strict_oof"]
    assert "historical_b17_weights" in oof
    assert "strict_oof_weights" in oof
    assert len(oof["strict_oof_weights"]) == 3
    # Check that weights are valid probability distribution (non-negative, sum to 1)
    w = oof["strict_oof_weights"]
    assert all(x >= 0 for x in w)
    assert abs(sum(w) - 1.0) < 1e-4
