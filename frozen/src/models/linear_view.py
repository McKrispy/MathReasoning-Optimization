"""MOD-M07b: L2 logistic regression on FEAT-G06 character-hash TF-IDF.

The classifier predicts whether one candidate is correct. Its inputs and loss
are the hashed text and the binary training label. It does not estimate a
process reward or a reference-model likelihood ratio.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from features.char_hash import CharHashTfidf, linear_scores, tfidf_matrix

TEXT_LOGISTIC_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}


class TextLogisticView:
    """One fold-local text view. IDF and the classifier see only the rows passed to fit."""

    def __init__(self) -> None:
        self.vectorizer = CharHashTfidf()
        self.classifier: LogisticRegression | None = None

    def fit(
        self,
        idf_counts: Sequence[Mapping[int, int]],
        classifier_counts: Sequence[Mapping[int, int]],
        labels: np.ndarray,
    ) -> "TextLogisticView":
        """IDF uses every supplied fit candidate. The classifier uses the hard-candidate rows only."""
        self.vectorizer.fit_counts(idf_counts)
        matrix = self.vectorizer.transform_counts(classifier_counts)
        self.classifier = LogisticRegression(**TEXT_LOGISTIC_PARAMS)
        self.classifier.fit(matrix, np.asarray(labels, dtype=np.int32).reshape(-1))
        return self

    def decision_function(self, counts: Sequence[Mapping[int, int]]) -> np.ndarray:
        if self.classifier is None or self.vectorizer.idf_ is None:
            raise RuntimeError("TextLogisticView.fit must run before scoring")
        matrix = self.vectorizer.transform_counts(counts)
        return self._scores(matrix)

    def _scores(self, matrix: sparse.csr_matrix) -> np.ndarray:
        coefficients = np.asarray(self.classifier.coef_, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(self.classifier.intercept_).reshape(-1)[0])
        manual = linear_scores(matrix, coefficients, intercept)
        framework = np.asarray(self.classifier.decision_function(matrix), dtype=np.float64).reshape(-1)
        if not np.allclose(manual, framework, atol=1e-6, rtol=1e-6):
            raise RuntimeError("sparse NumPy text scores disagree with the fitted logistic decision function")
        return manual

    def export_arrays(self) -> Dict[str, Any]:
        if self.classifier is None or self.vectorizer.idf_ is None:
            raise RuntimeError("nothing to export")
        return {
            "idf": self.vectorizer.idf_.astype(np.float32),
            "coefficients": np.asarray(self.classifier.coef_, dtype=np.float32).reshape(-1),
            "intercept": np.asarray(self.classifier.intercept_, dtype=np.float32).reshape(-1),
            "n_fit": np.asarray([self.vectorizer.n_fit_], dtype=np.int64),
            "dimensions": np.asarray([self.vectorizer.dimensions], dtype=np.int64),
        }


def score_exported(counts: Sequence[Mapping[int, int]], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Inference with exported IDF and coefficients only."""
    matrix = tfidf_matrix(counts, np.asarray(arrays["idf"], dtype=np.float64))
    intercept = float(np.asarray(arrays["intercept"]).reshape(-1)[0])
    return linear_scores(matrix, np.asarray(arrays["coefficients"], dtype=np.float64), intercept)
