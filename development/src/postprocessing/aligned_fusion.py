"""POST-P07: fusion whose fit loss uses the same mask, temperature and clusters as decode.

The historical fitter minimizes an unmasked candidate listwise loss at temperature 1.
Deployment then masks candidates and aggregates answer clusters at temperature 0.5.
This module fits nonnegative weights on the deployed cluster objective and decodes
with that same objective. rho = 0 selects the same cluster as a sum of softmax
probabilities; a positive rho down-weights repeated trajectories.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
    detect_parser_heading_conflict,
)
from postprocessing.soft_stacking import standardize_within_problem

import sys
from pathlib import Path

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore

MASK_PENALTY = 1e9
PROBABILITY_FLOOR = 1e-12
SCALE_FLOOR = 1e-6
SENTINEL_FLOOR = -1e8
OPTIMIZER_OPTIONS = {"ftol": 1e-7, "maxiter": 200}


def nested_attention_seeds(variant: int, seed: int, outer: int, inner: int) -> Dict[str, int]:
    """Contract seed stream. variant 0 is arm A and variant 1 is arm B."""
    warmup = 1_000_000 + 100_000 * int(variant) + 10_000 * int(seed) + 10 * int(outer) + int(inner)
    return {"warmup": int(warmup), "finetune": int(warmup + 1000)}


def b25_configs(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """D1, then D2, then D3, in the contract's listed temperature and rho order."""
    batch = contract["batches"]["EXP-B25"]
    configs: List[Dict[str, Any]] = []
    for arm_name in ("D1", "D2", "D3"):
        arm = batch["arms"][arm_name]
        for temperature in arm["temperatures"]:
            for rho in arm["rho"]:
                configs.append({
                    "arm": arm_name,
                    "normalization": arm["normalization"],
                    "temperature": float(temperature),
                    "rho": float(rho),
                    "name": (
                        f"{arm_name}/{arm['normalization']}/T={float(temperature):.2f}/rho={float(rho):.1f}"
                    ),
                })
    if len(configs) != int(batch["max_unique_inner_configurations"]):
        raise RuntimeError(
            f"B25 produced {len(configs)} configs, contract allows {batch['max_unique_inner_configurations']}"
        )
    if any(item["arm"] == "D2" and item["rho"] != 0.0 for item in configs):
        raise RuntimeError("D2 must keep rho at 0; D2 x D3 is not allowed")
    if any(item["arm"] == "D3" and item["normalization"] != "question_zscore" for item in configs):
        raise RuntimeError("D3 must keep question z-score normalization")
    return configs


def question_eligibility(bank: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[List[Optional[str]]]]:
    """Same eligible set as POST-P05 validity and heading-conflict masks."""
    eligible = np.ones((len(bank), 4), dtype=bool)
    parses: List[List[Optional[str]]] = []
    for i, row in enumerate(bank):
        texts = [candidate["text"] for candidate in row["candidates"]]
        parsed = [common.parse_answer(text) for text in texts]
        conflicts = [detect_parser_heading_conflict(text) for text in texts]
        valid = [item is not None for item in parsed]
        parses.append(parsed)
        if any(valid):
            for index, is_valid in enumerate(valid):
                if not is_valid:
                    eligible[i, index] = False
        safe = [is_valid and not conflict for is_valid, conflict in zip(valid, conflicts)]
        if any(safe):
            for index, conflict in enumerate(conflicts):
                if conflict:
                    eligible[i, index] = False
    return eligible, parses


def gold_clusters(parses: Sequence[Sequence[Optional[str]]], labels: np.ndarray) -> List[Optional[str]]:
    """Canonical answer of the first positive candidate. Positives share one parse."""
    labels = np.asarray(labels)
    gold: List[Optional[str]] = []
    for i, parsed in enumerate(parses):
        positive = np.flatnonzero(labels[i] >= 0.5)
        gold.append(None if positive.size == 0 else parsed[int(positive[0])])
    return gold


def usable_score_mask(scores: np.ndarray) -> np.ndarray:
    """Finite raw scores. Mask sentinels stay out of the center and the RMS."""
    arr = np.asarray(scores, dtype=np.float64)
    return np.isfinite(arr) & (arr > SENTINEL_FLOOR)


def center_finite_rows(scores: np.ndarray) -> np.ndarray:
    """Subtract the mean of the usable scores in each question."""
    arr = np.asarray(scores, dtype=np.float64)
    usable = usable_score_mask(arr)
    counts = usable.sum(axis=1, keepdims=True)
    summed = np.where(usable, arr, 0.0).sum(axis=1, keepdims=True)
    mean = np.divide(summed, np.maximum(counts, 1.0))
    return np.where(usable, arr - mean, arr)


def fit_global_rms(scores: np.ndarray, eligible: np.ndarray, rows: np.ndarray) -> float:
    used = np.asarray(scores, dtype=np.float64)[rows]
    centered = center_finite_rows(used)
    mask = np.asarray(eligible[rows], dtype=bool) & usable_score_mask(used)
    values = centered[mask]
    if values.size == 0:
        raise RuntimeError("global scale has no eligible finite scores in the fit rows")
    return float(max(np.sqrt(np.mean(values * values)), SCALE_FLOOR))


def apply_global_scale(scores: np.ndarray, rms: float) -> np.ndarray:
    return (center_finite_rows(scores) / max(float(rms), SCALE_FLOOR)).astype(np.float64)


def normalize_views(
    views: Sequence[np.ndarray],
    normalization: str,
    eligible: np.ndarray,
    rows: Optional[np.ndarray] = None,
    scales: Optional[Sequence[float]] = None,
) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    if normalization == "question_zscore":
        return [np.asarray(standardize_within_problem(view), dtype=np.float64) for view in views], None
    if normalization != "fold_OOF_global_scale":
        raise ValueError(f"unknown normalization {normalization}")
    if scales is None:
        if rows is None:
            rows = np.arange(views[0].shape[0])
        scales = [fit_global_rms(view, eligible, rows) for view in views]
    return [apply_global_scale(view, rms) for view, rms in zip(views, scales)], [float(item) for item in scales]


def blend(weights: Sequence[float], views: Sequence[np.ndarray]) -> np.ndarray:
    coeffs = np.asarray(weights, dtype=np.float64)
    total = float(coeffs.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("fusion weights must be finite and positive in sum")
    coeffs = coeffs / total
    stacked = np.zeros(views[0].shape, dtype=np.float64)
    for coeff, view in zip(coeffs, views):
        stacked += coeff * np.asarray(view, dtype=np.float64)
    return stacked


def build_cluster_table(
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pad each question to at most four answer clusters so the fit loss stays vectorized."""
    count = len(parses)
    cluster_index = np.full((count, 4, 4), -1, dtype=np.int64)
    cluster_count = np.zeros((count, 4), dtype=np.int64)
    gold_cluster = np.full(count, -1, dtype=np.int64)
    gold = gold_clusters(parses, labels)
    for row, parsed in enumerate(parses):
        buckets: Dict[str, List[int]] = {}
        for index in range(4):
            if not bool(eligible[row, index]) or parsed[index] is None:
                continue
            buckets.setdefault(parsed[index], []).append(index)
        for cluster_id, members in enumerate(buckets.values()):
            cluster_count[row, cluster_id] = len(members)
            cluster_index[row, cluster_id, : len(members)] = members
        if gold[row] is not None and gold[row] in buckets:
            gold_cluster[row] = list(buckets).index(gold[row])
    return {
        "cluster_index": cluster_index,
        "cluster_count": cluster_count,
        "gold_cluster": gold_cluster,
    }


def cluster_log_probabilities(
    scores: np.ndarray,
    table: Dict[str, np.ndarray],
    rows: np.ndarray,
    temperature: float,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-question log probability of the gold cluster. Missing clusters use the 1e-12 floor."""
    rows = np.asarray(rows, dtype=np.int64)
    used = np.asarray(scores, dtype=np.float64)[rows]
    index = table["cluster_index"][rows]
    counts = table["cluster_count"][rows]
    gold = table["gold_cluster"][rows]
    member_ok = index >= 0
    safe_index = np.clip(index, 0, 3)
    gathered = used[np.arange(rows.size)[:, None, None], safe_index]
    scaled = np.where(member_ok, gathered / float(temperature), -1e30)
    peak = np.max(scaled, axis=2)
    member_total = np.sum(np.exp(np.clip(scaled - peak[:, :, None], -80.0, 80.0)) * member_ok, axis=2)
    log_sum = np.where(member_ok.any(axis=2), peak + np.log(np.maximum(member_total, 1e-30)), -1e30)
    energy = log_sum - float(rho) * np.log(np.maximum(counts, 1).astype(np.float64))
    cluster_ok = counts > 0
    energy = np.where(cluster_ok, energy, -1e30)
    peak_energy = np.max(energy, axis=1, keepdims=True)
    weights = np.exp(np.clip(energy - peak_energy, -80.0, 80.0)) * cluster_ok
    total = np.sum(weights, axis=1)
    has_gold = gold >= 0
    gold_weight = weights[np.arange(rows.size), np.clip(gold, 0, 3)]
    probability = np.where(has_gold, gold_weight / np.maximum(total, 1e-30), PROBABILITY_FLOOR)
    probability = np.maximum(probability, PROBABILITY_FLOOR)
    return np.log(probability), has_gold


def choose_from_scores(
    scores: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    temperature: float,
    rho: float,
) -> np.ndarray:
    """Pick one candidate per question.

    rho = 0 calls the deployed mask and probability-sum aggregator, so a fixed
    blend keeps the current selection rule. A positive rho uses F directly.
    """
    if float(rho) == 0.0:
        masked = apply_validity_mask(scores, bank, penalty=MASK_PENALTY)
        masked = apply_conflict_penalty(masked, bank, penalty=MASK_PENALTY)
        return np.asarray(
            SoftmaxClusterAggregator(temperature=temperature).aggregate(bank, masked),
            dtype=np.int64,
        )
    choices = np.empty(scores.shape[0], dtype=np.int64)
    for row in range(scores.shape[0]):
        buckets: Dict[str, List[int]] = {}
        for index in range(4):
            if not bool(eligible[row, index]) or parses[row][index] is None:
                continue
            buckets.setdefault(parses[row][index], []).append(index)
        if not buckets:
            choices[row] = int(np.argmax(scores[row]))
            continue
        energies = {
            key: float(logsumexp(np.asarray(scores[row, members], dtype=np.float64) / temperature) - rho * np.log(len(members)))
            for key, members in buckets.items()
        }

        def cluster_rank(key: str) -> Tuple[float, float, int]:
            members = buckets[key]
            return (energies[key], max(float(scores[row, member]) for member in members), -min(members))

        best_key = max(buckets, key=cluster_rank)
        choices[row] = int(max(buckets[best_key], key=lambda member: (float(scores[row, member]), -member)))
    return choices


def project_weights(raw: Sequence[float]) -> np.ndarray:
    coeffs = np.maximum(0.0, np.asarray(raw, dtype=np.float64))
    total = float(coeffs.sum())
    if total <= 0.0 or not np.isfinite(coeffs).all():
        raise RuntimeError("fusion optimizer returned non-finite weights")
    return coeffs / total


def fit_aligned_weights(
    views: Sequence[np.ndarray],
    hard_mask: np.ndarray,
    table: Dict[str, np.ndarray],
    temperature: float,
    rho: float,
    rows: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit a convex combination on train-hard rows. Views are already normalized."""
    rows = np.asarray(rows, dtype=np.int64)
    hard_rows = rows[np.asarray(hard_mask, dtype=bool)[rows]]
    if hard_rows.size == 0:
        raise RuntimeError("aligned fusion fit has no hard questions")
    view_count = len(views)
    prepared = [np.asarray(view, dtype=np.float64) for view in views]
    absent = 0

    def objective(raw: Sequence[float]) -> float:
        nonlocal absent
        coeffs = np.maximum(0.0, np.asarray(raw, dtype=np.float64))
        total = float(coeffs.sum())
        if total <= 0.0 or not np.isfinite(coeffs).all():
            return 1e6
        log_prob, has_gold = cluster_log_probabilities(
            blend(coeffs / total, prepared), table, hard_rows, temperature, rho,
        )
        absent = int((~has_gold).sum())
        return float(np.mean(-log_prob))

    initial = np.full(view_count, 1.0 / view_count, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * view_count,
        constraints={"type": "eq", "fun": lambda raw: float(np.sum(raw) - 1.0)},
        options=OPTIMIZER_OPTIONS,
    )
    if (not bool(result.success)) or (not np.isfinite(result.x).all()) or (not np.isfinite(result.fun)):
        raise RuntimeError(
            f"aligned fusion optimizer failed: success={result.success} "
            f"message={result.message} fun={result.fun}"
        )
    weights = project_weights(result.x)
    objective(weights)
    return weights, {
        "loss": float(result.fun),
        "iterations": int(result.nit),
        "absent_gold_clusters": int(absent),
        "hard_questions": int(hard_rows.size),
    }


def score_aligned(
    views: Sequence[np.ndarray],
    labels: np.ndarray,
    hard_mask: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses: Sequence[Sequence[Optional[str]]],
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    weights: Sequence[float],
    temperature: float,
    rho: float,
    rows: np.ndarray,
) -> Dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    scores = blend(weights, views)
    sliced_bank = [bank[int(row)] for row in rows]
    sliced_parses = [parses[int(row)] for row in rows]
    choices = choose_from_scores(
        scores[rows], sliced_bank, sliced_parses, eligible[rows], temperature, rho,
    )
    hit_rows = labels[rows, choices] >= 0.5
    hard = np.asarray(hard_mask, dtype=bool)[rows]
    log_prob, has_gold = cluster_log_probabilities(scores, table, rows, temperature, rho)
    return {
        "hard_correct": int(hit_rows[hard].sum()),
        "hard_total": int(hard.sum()),
        "all_correct": int(hit_rows.sum()),
        "all_total": int(rows.size),
        "nll_sum": float(np.sum(-log_prob)),
        "nll_count": int(rows.size),
        "absent_gold_clusters": int((~has_gold).sum()),
        "choices": choices,
    }


def select_config(scored: Sequence[Dict[str, Any]]) -> int:
    """Highest hard correct, then all correct, then lowest NLL, then earlier declared arm."""
    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, float, int]:
        index, score = item
        return (
            int(score["hard_correct"]),
            int(score["all_correct"]),
            -float(score["nll_sum"]) / max(1, int(score["nll_count"])),
            -index,
        )

    best_index, _score = max(enumerate(scored), key=sort_key)
    return int(best_index)


def internal_gate_decision(
    reference: Sequence[Dict[str, int]],
    selected: Sequence[Dict[str, int]],
    minimum_gain: float,
    minimum_positive_folds: int,
) -> Dict[str, Any]:
    """Both the unweighted mean of fold hard rates and the pooled hard rate must clear the gain."""
    if len(reference) != len(selected) or len(reference) == 0:
        raise ValueError("reference and selected fold counts must match")

    def mean_hard(rows: Sequence[Dict[str, int]]) -> float:
        return float(np.mean([100.0 * row["hard_correct"] / row["hard_total"] for row in rows]))

    def pooled(rows: Sequence[Dict[str, int]], correct_key: str, total_key: str) -> Tuple[int, int]:
        correct = int(sum(row[correct_key] for row in rows))
        total = int(sum(row[total_key] for row in rows))
        return correct, total

    ref_hard = pooled(reference, "hard_correct", "hard_total")
    new_hard = pooled(selected, "hard_correct", "hard_total")
    ref_all = pooled(reference, "all_correct", "all_total")
    new_all = pooled(selected, "all_correct", "all_total")
    mean_gain = mean_hard(selected) - mean_hard(reference)
    pooled_gain = 100.0 * new_hard[0] / new_hard[1] - 100.0 * ref_hard[0] / ref_hard[1]
    positive = int(sum(
        selected[index]["hard_correct"] > reference[index]["hard_correct"]
        for index in range(len(reference))
    ))
    all_delta = int(new_all[0] - ref_all[0])
    passed = bool(
        mean_gain >= minimum_gain
        and pooled_gain >= minimum_gain
        and positive >= minimum_positive_folds
        and all_delta >= 0
    )
    return {
        "passed": passed,
        "mean_hard_gain_percentage_points": float(mean_gain),
        "pooled_hard_gain_percentage_points": float(pooled_gain),
        "positive_outer_folds": positive,
        "pooled_all_delta": all_delta,
        "reference_hard_mean_percent": mean_hard(reference),
        "selected_hard_mean_percent": mean_hard(selected),
        "reference_pooled_hard": {"correct": ref_hard[0], "total": ref_hard[1]},
        "selected_pooled_hard": {"correct": new_hard[0], "total": new_hard[1]},
        "reference_pooled_all": {"correct": ref_all[0], "total": ref_all[1]},
        "selected_pooled_all": {"correct": new_all[0], "total": new_all[1]},
    }
