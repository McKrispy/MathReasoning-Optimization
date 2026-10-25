"""Family-block shuffle checks for EXP-B11."""
from pathlib import Path
import sys
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b11_family_shuffle import family_column_index, shuffle_question_blocks


def test_columns_are_grouped_by_the_contract_family():
    contract = {
        "statistically_eligible_by_family": {
            "FEAT-G01": ["is_valid", "cluster_ratio"],
            "FEAT-G03": ["num_equations"],
        }
    }
    grouped = family_column_index(["num_equations", "is_valid", "cluster_ratio"], contract)
    assert grouped["FEAT-G03"] == [0]
    assert grouped["FEAT-G01"] == [1, 2]


def test_shuffle_moves_a_family_block_and_leaves_the_rest():
    features = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    shuffled = shuffle_question_blocks(features, [0, 1], np.array([1, 0]))
    assert np.array_equal(shuffled[0, :, :2], features[1, :, :2])
    assert np.array_equal(shuffled[1, :, :2], features[0, :, :2])
    assert np.array_equal(shuffled[:, :, 2], features[:, :, 2])


if __name__ == "__main__":
    test_columns_are_grouped_by_the_contract_family()
    test_shuffle_moves_a_family_block_and_leaves_the_rest()
    print("test_exp_b11 ok")
