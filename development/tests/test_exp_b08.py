"""Column-contract checks for EXP-B08. The full fit is the experiment script."""
from pathlib import Path
import sys

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b08_reasoning import load_contract, select_reasoning_columns


def test_reasoning_columns_follow_the_locked_contract():
    g01, g03 = select_reasoning_columns(load_contract())
    assert g01 == ["is_valid", "cluster_ratio", "is_unique_majority", "is_tied_majority"]
    assert "log_magnitude" not in g01
    assert "is_integer" not in g01
    assert "appears_in_question" not in g01
    assert len(g03) == 12
    assert "has_arithmetic_error" in g03
    assert "missing_condition_count" in g03
    assert "question_number_count" not in g03
    assert "correct_equation_ratio" not in g03


def test_morphology_cannot_enter_the_core():
    contract = {
        "gsm8k_morphology_columns_not_core": ["log_magnitude"],
        "statistically_eligible_by_family": {
            "FEAT-G01": ["is_valid", "log_magnitude"],
            "FEAT-G03": ["has_arithmetic_error"],
        },
    }
    g01, g03 = select_reasoning_columns(contract)
    assert g01 == ["is_valid"]
    assert g03 == ["has_arithmetic_error"]


if __name__ == "__main__":
    test_reasoning_columns_follow_the_locked_contract()
    test_morphology_cannot_enter_the_core()
    print("test_exp_b08 ok")
