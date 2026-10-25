"""POST-P01: Temperature scaling for candidate correctness probabilities.

Scores are independent logits because more than one candidate can be correct.
The calibrated confidence is sigmoid(logit / T), fit by validation binary NLL.
"""
from typing import Any, Dict, List, Tuple
import numpy as np


def sigmoid_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.clip(np.asarray(logits, dtype=np.float64) / float(temperature), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def binary_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(target * np.log(probs) + (1.0 - target) * np.log(1.0 - probs)))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Equal-width ECE. Empty bins are omitted from the weighted sum."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.shape != target.shape:
        raise ValueError("probabilities and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = int(probs.size)
    error = 0.0
    rows: List[Dict[str, Any]] = []
    for bin_id in range(n_bins):
        low, high = edges[bin_id], edges[bin_id + 1]
        if bin_id == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(target[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
        rows.append({
            "bin": bin_id,
            "low": float(low),
            "high": float(high),
            "count": count,
            "confidence": confidence,
            "accuracy": accuracy,
        })
    return float(error), rows


class TemperatureScaler:
    """One positive temperature for every candidate logit."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Minimize binary NLL over T. Accuracy is not the fitting objective."""
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        log_grid = np.linspace(np.log(0.05), np.log(50.0), 97)

        def objective(log_temperature: float) -> float:
            return binary_nll(sigmoid_probs(logits, float(np.exp(log_temperature))), labels)

        best_log = min((float(value) for value in log_grid), key=objective)
        refined = np.linspace(best_log - 0.12, best_log + 0.12, 97)
        best_log = min((float(value) for value in refined), key=objective)
        self.temperature = float(np.exp(best_log))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return correctness probabilities, one per candidate. They need not sum to 1."""
        return sigmoid_probs(logits, self.temperature).astype(np.float32)
