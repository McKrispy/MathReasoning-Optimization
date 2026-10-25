"""EXP-B24 gate arithmetic and, after the run, the audit's validation lock."""
import json
import sys
from pathlib import Path

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b24_char_ngram import internal_gain_percentage_points

AUDIT_PATH = Path(__file__).resolve().parents[1] / "runs" / "audit_logs" / "exp_b24_char_ngram_audit.json"


def test_one_percentage_point_is_the_internal_threshold():
    old = [(50, 100), (50, 100)]
    short = [(50, 100), (51, 100)]
    enough = [(50, 100), (52, 100)]
    assert internal_gain_percentage_points(old, short) == 0.5
    assert internal_gain_percentage_points(old, enough) == 1.0


def test_exp_b24_audit_records_whether_validation_was_opened():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    gate = audit["internal_gate"]
    assert gate["minimum_percentage_points"] == 1.0
    assert len(gate["folds"]) == 5
    assert audit["test_references_opened"] is False
    assert audit["written_to_model_bundle"] is False
    if audit["verdict"] == "STOPPED_INTERNAL_GATE":
        assert gate["passed"] is False
        assert audit["validation_queries"] == 0
        assert audit["validation_confirmation"] is None
    else:
        assert gate["passed"] is True
        assert audit["validation_queries"] > 0
