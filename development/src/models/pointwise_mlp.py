"""MOD-M02: Pointwise candidate verifier (evaluates candidates independently).

Used to address Research Question 3 (evaluating solutions independently vs in comparison).
"""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


class PointwiseVerifier:
    """Evaluates candidates strictly in isolation without cross-candidate awareness."""

    def __init__(self, c_reg: float = 10.0, max_iter: int = 500, random_state: int = 42):
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(C=c_reg, solver='lbfgs', max_iter=max_iter, random_state=random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PointwiseVerifier":
        """Fit model on flattened candidate features [N*4, D] and labels [N*4]."""
        if x.ndim == 3:
            n, c, d = x.shape
            x_flat = x.reshape(n * c, d)
            y_flat = y.reshape(-1)
        else:
            x_flat = x
            y_flat = y.reshape(-1)
        self.model.fit(x_flat, y_flat)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Output candidate probabilities [N, 4]."""
        if x.ndim == 3:
            n, c, d = x.shape
            x_flat = x.reshape(n * c, d)
            probs = self.model.predict_proba(x_flat)[:, 1].reshape(n, c)
            return probs.astype(np.float32)
        else:
            return self.model.predict_proba(x)[:, 1].astype(np.float32)

    def predict_rank(self, x: np.ndarray) -> np.ndarray:
        """Pick best candidate index per question [N, 4, D] -> [N]."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)
