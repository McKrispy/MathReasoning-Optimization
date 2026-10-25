"""Base classes and pipeline for feature extraction."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for all candidate feature extractors."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> "BaseFeatureExtractor":
        """Fit extractor on training set if needed (e.g. SVD, scalers). Default is stateless."""
        return self

    @abstractmethod
    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features.
        
        Args:
            bank: List of questions with 4 candidates each.
            arrays: Dictionary with 'embeddings' [N, 4, 1024].
            
        Returns:
            np.ndarray of shape [N, 4, D] with dtype float32.
        """
        pass

    def fit_transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        return self.fit(bank, arrays).transform(bank, arrays)


class FeaturePipeline:
    """Combines multiple feature extractors and concatenates outputs along the last dimension."""

    def __init__(self, extractors: List[BaseFeatureExtractor]):
        self.extractors = extractors

    def fit(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> "FeaturePipeline":
        for ext in self.extractors:
            ext.fit(bank, arrays)
        return self

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        features = [ext.transform(bank, arrays) for ext in self.extractors]
        if not features:
            n = len(bank)
            return np.zeros((n, 4, 0), dtype=np.float32)
        return np.concatenate(features, axis=-1).astype(np.float32)

    def fit_transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        return self.fit(bank, arrays).transform(bank, arrays)
