"""Gate rules for EXP-B10. The tree fit is the experiment script."""
from pathlib import Path
import sys
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from postprocessing.aggregation import apply_disagreement_gate
from experiments.exp_b10_fusion import gate_specs, select_gate


def test_agreement_and_confidence_rules():
    m03 = np.array([0, 1, 2])
    m04 = np.array([0, 3, 1])
    conf03 = np.array([0.9, 0.4, 0.8])
    conf04 = np.array([0.2, 0.7, 0.6])
    assert np.array_equal(apply_disagreement_gate(m03, m04, conf03, conf04, "m03_only"), m03)
    assert np.array_equal(apply_disagreement_gate(m03, m04, conf03, conf04, "m04_only"), m04)
    higher = apply_disagreement_gate(m03, m04, conf03, conf04, "higher_confidence")
    assert np.array_equal(higher, np.array([0, 3, 2]))
    low = apply_disagreement_gate(m03, m04, conf03, conf04, "m04_when_m03_below", 0.5)
    assert np.array_equal(low, np.array([0, 3, 2]))
    tied = apply_disagreement_gate(m03, m04, np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]), "higher_confidence")
    assert np.array_equal(tied, m03)


def test_selection_keeps_m03_when_fusion_only_ties():
    rows = [
        {"name": "m03_only", "threshold": None, "priority": 0, "val_hard": {"correct": 294}, "val_all": {"correct": 737}},
        {"name": "m04_only", "threshold": None, "priority": 1, "val_hard": {"correct": 294}, "val_all": {"correct": 737}},
        {"name": "higher_confidence", "threshold": None, "priority": 2, "val_hard": {"correct": 300}, "val_all": {"correct": 736}},
    ]
    chosen = select_gate(rows, m03_val_all_correct=737)
    assert chosen["name"] == "m03_only"
    assert [spec["threshold"] for spec in gate_specs() if spec["name"] == "m04_when_m03_below"] == [0.5, 0.6, 0.7, 0.8, 0.9]


if __name__ == "__main__":
    test_agreement_and_confidence_rules()
    test_selection_keeps_m03_when_fusion_only_ties()
    print("test_exp_b10 ok")
