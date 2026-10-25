"""FEAT-G04a: per-question centered embeddings.

d_{i,c} = ê_{i,c} - mean_k ê_{i,k}, with ê the L2-normalized embedding.
The only fitted quantity is one global scale that gives the centered vectors
unit mean norm on train_all.
"""
import numpy as np


def centered_embeddings(normalized: np.ndarray) -> np.ndarray:
    """[N, 4, D] L2-normalized embeddings -> [N, 4, D] within-question differences."""
    if normalized.ndim != 3:
        raise ValueError(f"expected [N, 4, D], got {normalized.shape}")
    return (normalized - normalized.mean(axis=1, keepdims=True)).astype(np.float32)


def fit_scale(centered_train: np.ndarray) -> float:
    norms = np.linalg.norm(centered_train, axis=-1)
    mean_norm = float(norms.mean())
    if mean_norm <= 0:
        raise ValueError("centered vectors have zero mean norm")
    return 1.0 / mean_norm
