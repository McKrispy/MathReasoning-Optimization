"""Fast checks for temperature scaling and answer-cluster aggregation."""
from pathlib import Path
import sys
import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from postprocessing.aggregation import WeightedSelfConsistencyAggregator
from postprocessing.calibration import TemperatureScaler, binary_nll, sigmoid_probs
from experiments.exp_b09_calibration import GAMMA_GRID, select_gamma


def test_temperature_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=1.5, size=(4000, 4))
    true_t = 2.5
    probabilities = sigmoid_probs(logits, true_t)
    labels = rng.random(probabilities.shape) < probabilities
    fitted = TemperatureScaler().fit(logits, labels)
    assert 2.0 < fitted.temperature < 3.1, fitted.temperature
    fitted_nll = binary_nll(fitted.transform(logits), labels)
    unscaled_nll = binary_nll(sigmoid_probs(logits, 1.0), labels)
    assert fitted_nll < unscaled_nll


def test_gamma_zero_is_majority_and_large_gamma_is_best_of_n():
    bank = [{
        "candidates": [
            {"text": "#### 2"},
            {"text": "#### 3"},
            {"text": "#### 2"},
            {"text": "#### 5"},
        ]
    }]
    probabilities = np.array([[0.2, 0.9, 0.3, 0.1]], dtype=np.float32)
    majority = WeightedSelfConsistencyAggregator(gamma=0.0).aggregate(bank, probabilities)
    best = WeightedSelfConsistencyAggregator(gamma=20.0).aggregate(bank, probabilities)
    assert majority == [2]
    assert best == [1]


def test_gamma_grid_covers_the_plan_interval():
    assert GAMMA_GRID[0] == 0.0
    assert GAMMA_GRID[-1] == 5.0
    assert all(0.0 <= value <= 5.0 for value in GAMMA_GRID)


def test_gamma_selection_rejects_a_val_all_drop():
    rows = [
        {"gamma": 0.0, "val_hard": {"correct": 300}, "val_all": {"correct": 700}},
        {"gamma": 1.0, "val_hard": {"correct": 310}, "val_all": {"correct": 730}},
        {"gamma": 5.0, "val_hard": {"correct": 312}, "val_all": {"correct": 729}},
    ]
    chosen = select_gamma(rows, argmax_val_all_correct=730)
    assert chosen["gamma"] == 1.0


if __name__ == "__main__":
    test_temperature_recovers_a_known_scale()
    test_gamma_zero_is_majority_and_large_gamma_is_best_of_n()
    test_gamma_grid_covers_the_plan_interval()
    test_gamma_selection_rejects_a_val_all_drop()
    print("test_exp_b09 ok")
