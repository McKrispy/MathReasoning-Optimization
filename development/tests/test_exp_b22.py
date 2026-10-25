"""Checks for EXP-B22 that do not refit the tree.

The audit assertions run after exp_b22_g05_r5.py. They check the recorded gate,
the 150-d input, and that this batch did not revive the rejected linear view.
"""
import json
from pathlib import Path
import sys

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b22_g05_r5 import NEW_DIM, retention_decision

RUNS = Path(__file__).resolve().parents[1] / "runs"
AUDIT_PATH = RUNS / "audit_logs" / "exp_b22_g05_r5_audit.json"
CONTRACT_PATH = RUNS / "audit_logs" / "private_configuration_omitted.json"


def test_contract_dimensions():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["column_contract"]
    assert len(contract["structured_22"]) == 22
    assert contract["structured_22"][:16] == contract["structured_16"]
    assert contract["r5_with_g05_total_dimensions"] == NEW_DIM == 150


def test_retention_rejects_a_small_tree_only_gain():
    decision = retention_decision([2, 2, 2], [1, 1, 1], ensemble_hard=302, ensemble_all=744)
    assert decision["verdict"] == "REJECTED"
    assert decision["gates"]["rule1_paired_hard_mean_ge_3_55"]["passed"] is False
    assert decision["gates"]["rule4_ensemble_hard_vs_historical_ge_4"]["passed"] is False


def test_retention_accepts_only_when_every_rule_passes():
    decision = retention_decision([4, 4, 4], [1, 0, 2], ensemble_hard=304, ensemble_all=743)
    assert decision["verdict"] == "RETAINED"
    assert all(item["passed"] for item in decision["gates"].values())


def test_exp_b22_audit_records_the_pre_registered_comparison():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["input_dimensions"]["total"] == 150
    assert audit["b21_linear_view_included"] is False
    assert audit["fit_rows"] == "train_all"
    assert audit["assets"]["written_to_model_bundle"] is False
    assert audit["validation_configuration_ledger"]["temperature_or_hyperparameter_search_on_validation"] == 0
    assert audit["final_ensemble"]["historical_old_tree"]["val_hard"] == 300
    assert audit["final_ensemble"]["historical_old_tree"]["val_all"] == 743
    assert audit["final_ensemble"]["strict_old_tree"]["val_hard"] == 295
    assert audit["retention_gate"]["verdict"] in {"RETAINED", "REJECTED"}
    assert audit["official_score"]["correct"] == audit["final_ensemble"]["new_tree"]["val_all"]
    assert len(audit["standalone_argmax"]["new_r5"]) == 3
