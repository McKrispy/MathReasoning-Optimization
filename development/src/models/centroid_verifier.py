"""FEAT-G07 / MOD-M08: two train-only centroid scores on supplied candidate embeddings.

C0 is one weighted mean per class and a squared-distance difference. C1 keeps at
most four weighted prototypes per class and scores a shared radial-basis mixture.
Both use the same question-class weights: inside one question and one label, the
valid candidates share weight 1. Invalid candidates are left out of the fit.
Scoring still returns one value for every candidate.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

ARMS: Tuple[str, ...] = ("C0", "C1")
C1_CAP = 4
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
TAU_FLOOR = 1e-6


class InvalidCentroidFit(RuntimeError):
    """A class has no valid fit candidates, or every prototype came back empty."""


def question_class_weights(
    labels: np.ndarray,
    parses: Sequence[Sequence[Optional[str]]],
    hard: np.ndarray,
    rows: np.ndarray,
) -> Dict[str, object]:
    """Collect valid Hard candidates. Each question contributes total weight 1 to a class."""
    labels = np.asarray(labels)
    hard = np.asarray(hard, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    points = {0: [], 1: []}
    weights = {0: [], 1: []}
    questions = {0: 0, 1: 0}
    skipped_unparsed = {0: 0, 1: 0}
    for row in rows:
        row = int(row)
        if not bool(hard[row]):
            continue
        for class_id in (0, 1):
            members = []
            for candidate in range(4):
                positive = float(labels[row, candidate]) >= 0.5
                if positive != (class_id == 1):
                    continue
                if parses[row][candidate] is None:
                    skipped_unparsed[class_id] += 1
                    continue
                members.append(candidate)
            if not members:
                continue
            questions[class_id] += 1
            share = 1.0 / float(len(members))
            for candidate in members:
                points[class_id].append((row, candidate))
                weights[class_id].append(share)
    return {
        "points": points,
        "weights": weights,
        "questions": questions,
        "skipped_unparsed": skipped_unparsed,
    }


def _class_matrix(
    embeddings: np.ndarray,
    packed: Dict[str, np.ndarray],
    class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pairs = packed["points"][class_id]
    if not pairs:
        raise InvalidCentroidFit(f"class {class_id} has no valid hard candidates in this fit partition")
    rows = np.asarray([item[0] for item in pairs], dtype=np.int64)
    candidates = np.asarray([item[1] for item in pairs], dtype=np.int64)
    vectors = np.asarray(embeddings[rows, candidates], dtype=np.float64)
    weights = np.asarray(packed["weights"][class_id], dtype=np.float64)
    if vectors.ndim != 2 or weights.shape[0] != vectors.shape[0] or float(weights.sum()) <= 0.0:
        raise InvalidCentroidFit(f"class {class_id} weights are empty")
    return vectors, weights


def _distinct_rows(vectors: np.ndarray) -> int:
    return int(np.unique(vectors, axis=0).shape[0])


def _weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(vectors, axis=0, weights=weights)


def _squared_distances(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance of every vector to every center."""
    left = np.sum(vectors * vectors, axis=1, keepdims=True)
    right = np.sum(centers * centers, axis=1)[None, :]
    dots = vectors @ centers.T
    return np.maximum(left + right - 2.0 * dots, 0.0)


def _drop_empty(centers: np.ndarray, assignments: np.ndarray, requested: int) -> np.ndarray:
    present = np.zeros(requested, dtype=bool)
    counted = np.bincount(np.asarray(assignments, dtype=np.int64), minlength=requested)
    present[: counted.shape[0]] = counted > 0
    kept = centers[present]
    if kept.shape[0] == 0:
        raise InvalidCentroidFit("every KMeans prototype was empty")
    return np.asarray(kept, dtype=np.float64)


def _fit_prototypes(vectors: np.ndarray, weights: np.ndarray, cap: int) -> Tuple[np.ndarray, int, int]:
    distinct = _distinct_rows(vectors)
    requested = int(min(cap, distinct, vectors.shape[0]))
    if requested <= 0:
        raise InvalidCentroidFit("prototype fit has no distinct vectors")
    if requested == 1:
        return _weighted_mean(vectors, weights)[None, :], requested, 0
    model = KMeans(
        n_clusters=requested,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        algorithm="lloyd",
    )
    model.fit(vectors, sample_weight=weights)
    iterations = int(np.max(np.asarray(model.n_iter_)))
    centers = _drop_empty(np.asarray(model.cluster_centers_, dtype=np.float64), model.labels_, requested)
    return centers, requested, iterations


def _shared_tau(
    correct: np.ndarray,
    correct_weights: np.ndarray,
    correct_centers: np.ndarray,
    wrong: np.ndarray,
    wrong_weights: np.ndarray,
    wrong_centers: np.ndarray,
) -> float:
    correct_gap = _squared_distances(correct, correct_centers).min(axis=1)
    wrong_gap = _squared_distances(wrong, wrong_centers).min(axis=1)
    correct_mean = float(np.average(correct_gap, weights=correct_weights))
    wrong_mean = float(np.average(wrong_gap, weights=wrong_weights))
    return float(max(0.5 * (correct_mean + wrong_mean), TAU_FLOOR))


def _counts(packed: Dict[str, np.ndarray], vectors: Dict[int, np.ndarray]) -> Dict[str, Dict[str, int]]:
    return {
        "questions": {"wrong": int(packed["questions"][0]), "correct": int(packed["questions"][1])},
        "candidates": {"wrong": int(vectors[0].shape[0]), "correct": int(vectors[1].shape[0])},
        "distinct_vectors": {"wrong": _distinct_rows(vectors[0]), "correct": _distinct_rows(vectors[1])},
        "skipped_unparsed": {
            "wrong": int(packed["skipped_unparsed"][0]),
            "correct": int(packed["skipped_unparsed"][1]),
        },
    }


def fit_centroids(
    arm: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    parses: Sequence[Sequence[Optional[str]]],
    hard: np.ndarray,
    rows: np.ndarray,
) -> Dict[str, object]:
    """Fit C0 or C1 on the hard valid candidates inside rows."""
    if arm not in ARMS:
        raise ValueError(f"unknown centroid arm {arm}")
    started = time.perf_counter()
    packed = question_class_weights(labels, parses, hard, rows)
    vectors = {}
    weights = {}
    for class_id in (0, 1):
        vectors[class_id], weights[class_id] = _class_matrix(embeddings, packed, class_id)
    if arm == "C0":
        correct = _weighted_mean(vectors[1], weights[1])[None, :]
        wrong = _weighted_mean(vectors[0], weights[0])[None, :]
        tau = None
        requested = {"correct": 1, "wrong": 1}
        iterations = {"correct": 0, "wrong": 0}
    else:
        correct, requested_correct, iter_correct = _fit_prototypes(vectors[1], weights[1], C1_CAP)
        wrong, requested_wrong, iter_wrong = _fit_prototypes(vectors[0], weights[0], C1_CAP)
        tau = _shared_tau(vectors[1], weights[1], correct, vectors[0], weights[0], wrong)
        requested = {"correct": requested_correct, "wrong": requested_wrong}
        iterations = {"correct": iter_correct, "wrong": iter_wrong}
    return {
        "arm": arm,
        "correct": np.asarray(correct, dtype=np.float64),
        "wrong": np.asarray(wrong, dtype=np.float64),
        "tau": tau,
        "requested_k": requested,
        "actual_k": {"correct": int(correct.shape[0]), "wrong": int(wrong.shape[0])},
        "kmeans_iterations": iterations,
        "class_counts": _counts(packed, vectors),
        "fit_seconds": float(time.perf_counter() - started),
    }


def score_centroids(model: Dict[str, object], embeddings: np.ndarray) -> np.ndarray:
    """Higher means closer to the correct class. One score per candidate."""
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != 4:
        raise ValueError(f"expected embeddings [N, 4, D], got {array.shape}")
    count, _, width = array.shape
    flat = array.reshape(count * 4, width)
    correct = np.asarray(model["correct"], dtype=np.float64)
    wrong = np.asarray(model["wrong"], dtype=np.float64)
    if model["arm"] == "C0":
        scores = _squared_distances(flat, wrong)[:, 0] - _squared_distances(flat, correct)[:, 0]
    elif model["arm"] == "C1":
        tau = float(model["tau"])
        if tau <= 0.0:
            raise ValueError("C1 tau must be positive")
        correct_mix = logsumexp(-_squared_distances(flat, correct) / tau, axis=1) - np.log(correct.shape[0])
        wrong_mix = logsumexp(-_squared_distances(flat, wrong) / tau, axis=1) - np.log(wrong.shape[0])
        scores = correct_mix - wrong_mix
    else:
        raise ValueError(f"unknown centroid arm {model['arm']}")
    return scores.reshape(count, 4)


def linear_c0_expansion(vectors: np.ndarray, correct: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """Closed form 2 x·(mu_correct-mu_wrong) + ||mu_wrong||^2 - ||mu_correct||^2."""
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong, dtype=np.float64).reshape(-1)
    direction = correct - wrong
    flat = np.asarray(vectors, dtype=np.float64).reshape(-1, direction.shape[0])
    offset = float(np.dot(wrong, wrong) - np.dot(correct, correct))
    return 2.0 * flat @ direction + offset
