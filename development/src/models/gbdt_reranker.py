"""MOD-M04: Gradient Boosted Decision Tree Listwise / Pointwise Reranker."""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class GBDTReranker:
    """Decision tree based reranker using engineered features."""

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ):
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GBDTReranker":
        """Fit GBDT on flattened or [N, 4, D] features and binary labels."""
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
        """Predict correctness probabilities for candidate groups [N, 4, D] -> [N, 4]."""
        if x.ndim == 3:
            n, c, d = x.shape
            x_flat = x.reshape(n * c, d)
            probs = self.model.predict_proba(x_flat)[:, 1].reshape(n, c)
            return probs
        else:
            return self.model.predict_proba(x)[:, 1]

    def predict_rank(self, x: np.ndarray) -> np.ndarray:
        """Select best candidate index for each question [N, 4, D] -> [N]."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)
