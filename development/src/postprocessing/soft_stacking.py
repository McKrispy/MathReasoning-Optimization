"""POST-P06: Cross-View Soft Logit Stacking & Multi-Model Heterogeneous Ensembling.

Provides within-problem logit standardization, training-set OOF weight optimization,
and soft convex combination ensembling across heterogeneous inductive biases.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"

if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
)


def standardize_within_problem(logits: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize logits within each question across the 4 candidates:
    
    tilde{s}_{i,c} = (s_{i,c} - mu_i) / max(sigma_i, eps)
    
    This eliminates arbitrary scale disparities across heterogeneous model families
    (e.g. neural nets, decision trees, logistic regression) while preserving
    intra-problem rank order and relative margins.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {arr.shape}")
    mu = np.mean(arr, axis=-1, keepdims=True)
    sigma = np.std(arr, axis=-1, keepdims=True)
    return ((arr - mu) / np.maximum(sigma, eps)).astype(np.float32)


def fit_convex_combination_weights(
    view_oofs: Sequence[np.ndarray],
    labels: np.ndarray,
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit convex combination weights w on training set OOF standardized logits.
    
    Minimizes Listwise Cross-Entropy Loss:
        L(w) = - 1/N sum_i log( sum_{c in C_i^+} exp(s_c) / sum_{j=0}^3 exp(s_j) )
    subject to w_m >= 0, sum w_m = 1.
    
    Strictly trained on training set OOF predictions to prevent validation leakage.
    """
    m = len(view_oofs)
    if m == 0:
        raise ValueError("Must provide at least one view")
    if m == 1:
        return np.array([1.0], dtype=np.float64)

    y = np.asarray(labels, dtype=np.float64)
    if subset_mask is not None:
        mask = np.asarray(subset_mask, dtype=bool)
        oofs_used = [v[mask] for v in view_oofs]
        y_used = y[mask]
    else:
        oofs_used = list(view_oofs)
        y_used = y

    # Optimize weights on simplex
    def listwise_loss(w):
        w_norm = np.array(w, dtype=np.float64)
        w_norm = w_norm / max(1e-12, float(np.sum(w_norm)))
        stacked = sum(w_norm[i] * oofs_used[i] for i in range(m))
        shift = stacked - np.max(stacked, axis=-1, keepdims=True)
        exp_s = np.exp(shift)
        num = np.sum(exp_s * y_used, axis=-1)
        den = np.sum(exp_s, axis=-1)
        return -float(np.mean(np.log(np.maximum(num / np.maximum(den, 1e-12), 1e-12))))

    init_w = [1.0 / m] * m
    bounds = [(0.0, 1.0)] * m
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        listwise_loss,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-7, "maxiter": 200},
    )

    w_opt = np.maximum(0.0, res.x)
    return (w_opt / np.sum(w_opt)).astype(np.float64)


class SoftStackingEnsemble:
    """POST-P06 Heterogeneous Soft Stacking Ensemble.
    
    Standardizes input logits within each question, computes convex combination:
        s^{ens}_{i,c} = sum_m w_m tilde{s}_{i,c}^{(m)}
    and chains validity masking and Softmax cluster aggregation.
    """

    def __init__(
        self,
        weights: Sequence[float],
        temperature: float = 1.0,
        apply_mask: bool = True,
        apply_de_mining: bool = True,
    ):
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative")
        self.weights = w / np.sum(w)
        self.temperature = float(temperature)
        self.apply_mask = apply_mask
        self.apply_de_mining = apply_de_mining
        self.aggregator = SoftmaxClusterAggregator(temperature=self.temperature)

    def blend_logits(self, view_logits: Sequence[np.ndarray]) -> np.ndarray:
        """Standardize each view within-problem and compute weighted sum."""
        if len(view_logits) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} views, got {len(view_logits)}")
        stacked = np.zeros_like(view_logits[0], dtype=np.float32)
        for i, logits in enumerate(view_logits):
            stacked += float(self.weights[i]) * standardize_within_problem(logits)
        return stacked

    def predict(
        self,
        view_logits: Sequence[np.ndarray],
        bank: Sequence[Dict[str, Any]],
    ) -> List[int]:
        """Compute stacked predictions for the problem bank."""
        blended = self.blend_logits(view_logits)
        if self.apply_mask:
            blended = apply_validity_mask(blended, bank, penalty=1e9)
        if self.apply_de_mining:
            blended = apply_conflict_penalty(blended, bank, penalty=1e9)
        return self.aggregator.aggregate(bank, blended)
