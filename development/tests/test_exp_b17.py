"""Unit tests for EXP-B17 cross-view soft stacking (POST-P06)."""
from pathlib import Path
import sys

import numpy as np
import pytest

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / "DASE7506-MP2" / "code"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from postprocessing.soft_stacking import (
    SoftStackingEnsemble,
    fit_convex_combination_weights,
    standardize_within_problem,
)


def test_standardize_within_problem_properties():
    # Test shape and basic mean/std properties
    logits = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 10.0, 10.0, 10.0],  # Constant row
        [-5.0, 0.0, 5.0, 10.0],
    ], dtype=np.float32)

    std_logits = standardize_within_problem(logits)
    assert std_logits.shape == logits.shape

    # Row 0: mean 0, std 1
    assert np.allclose(np.mean(std_logits[0]), 0.0, atol=1e-5)
    assert np.allclose(np.std(std_logits[0]), 1.0, atol=1e-5)

    # Row 1: constant row should not produce NaNs or infinities
    assert np.all(np.isfinite(std_logits[1]))
    assert np.allclose(std_logits[1], 0.0, atol=1e-4)

    # Shift-invariance: adding a constant C to all logits of a problem should yield identical output
    shifted = logits + 100.0
    std_shifted = standardize_within_problem(shifted)
    assert np.allclose(std_logits, std_shifted, atol=1e-5)


def test_fit_convex_combination_weights():
    # Synthetic dataset with 20 questions
    n = 20
    # True label: candidate 0 is always correct
    labels = np.zeros((n, 4), dtype=np.float32)
    labels[:, 0] = 1.0

    # View 1: very strong signal (candidate 0 has high logit)
    view_1 = np.zeros((n, 4), dtype=np.float32)
    view_1[:, 0] = 3.0

    # View 2: random noise
    np.random.seed(42)
    view_2 = np.random.randn(n, 4).astype(np.float32)

    std_v1 = standardize_within_problem(view_1)
    std_v2 = standardize_within_problem(view_2)

    weights = fit_convex_combination_weights([std_v1, std_v2], labels)
    assert len(weights) == 2
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    # View 1 should receive nearly all the weight
    assert weights[0] > 0.85
    assert weights[1] < 0.15


def test_soft_stacking_ensemble_end_to_end():
    bank = [
        {
            "id": "q1",
            "candidates": [
                {"text": "Answer is #### 10"},
                {"text": "Answer is #### 20"},
                {"text": "Answer is #### 30"},
                {"text": "Invalid option"},
            ],
        }
    ]
    # View 1 favors candidate 0 (answer 10)
    v1 = np.array([[3.0, 1.0, 0.0, -5.0]], dtype=np.float32)
    # View 2 also favors candidate 0
    v2 = np.array([[2.0, 0.5, 0.0, -10.0]], dtype=np.float32)

    ensemble = SoftStackingEnsemble(weights=[0.6, 0.4], temperature=1.0)
    choices = ensemble.predict([v1, v2], bank)
    assert len(choices) == 1
    assert choices[0] == 0
