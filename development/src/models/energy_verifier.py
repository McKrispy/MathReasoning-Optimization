"""MOD-M09: Bradley-Terry energy losses on the Arm A attention module.

Scores stay the network outputs. Energy is E = -s, so a higher score is a lower
energy. E1 averages softplus(E_positive - E_negative) inside each mixed question,
then across those questions. E2 adds the original per-question summed BCE.
Questions with no positive/negative pair contribute nothing to the BT term.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.exp_b07_cross_attn import HPARAMS
from models.cross_attn_net import CrossCandidateAttentionModule, state_dict_to_numpy
from postprocessing.aligned_fusion import score_aligned

SELECTABLE_ARMS: Tuple[str, ...] = ("E1", "E2")
VARIANTS = {"E0": 0, "E1": 2, "E2": 3}
DIAGNOSTIC_TEMPERATURES: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)


def bradley_terry_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
    """Mean over mixed questions of the mean pair softplus(s_negative - s_positive)."""
    if scores.ndim != 2 or labels.shape != scores.shape or scores.shape[1] != 4:
        raise ValueError(f"expected scores/labels [B, 4], got {tuple(scores.shape)} / {tuple(labels.shape)}")
    positive = labels >= 0.5
    pair = positive[:, :, None] & ~positive[:, None, :]
    pair_count = pair.sum(dim=(1, 2))
    gaps = F.softplus(scores[:, None, :] - scores[:, :, None])
    per_question = (gaps * pair.to(gaps.dtype)).sum(dim=(1, 2)) / pair_count.clamp(min=1).to(gaps.dtype)
    mixed = pair_count > 0
    info = {
        "mixed_questions": int(mixed.sum().item()),
        "pairs": int(pair_count.sum().item()),
        "questions": int(scores.shape[0]),
    }
    if not bool(mixed.any()):
        return None, info
    return per_question[mixed].mean(), info


def summed_bce(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of sum_4 BCE(sigmoid(s), y). sigmoid(s) is sigmoid(-E)."""
    per_row = F.binary_cross_entropy_with_logits(scores, labels, reduction="none").sum(dim=-1)
    return per_row.mean()


def energy_batch_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    arm: str,
) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
    """E1 is BT only. E2 is BT plus BCE. A batch with no pairs has no E1 loss."""
    if arm not in SELECTABLE_ARMS:
        raise ValueError(f"unknown energy arm {arm}")
    bt, info = bradley_terry_loss(scores, labels)
    info["all_positive_batch"] = int(info["mixed_questions"] == 0)
    if arm == "E1":
        return bt, info
    bce = summed_bce(scores, labels)
    loss = bce if bt is None else bt + bce
    return loss, info


def label_group_counts(labels: np.ndarray) -> Dict[str, int]:
    """Static pair counts for one fit matrix. Not multiplied by epochs."""
    labels = np.asarray(labels)
    positive = labels >= 0.5
    negative = ~positive
    pair_count = positive.sum(axis=1) * negative.sum(axis=1)
    mixed = pair_count > 0
    return {
        "questions": int(labels.shape[0]),
        "mixed_questions": int(mixed.sum()),
        "all_positive_questions": int(positive.all(axis=1).sum()),
        "pairs": int(pair_count.sum()),
    }


def _run_phase(
    model: CrossCandidateAttentionModule,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    arm: str,
) -> Dict[str, int]:
    """B12 batching. E1 does not step on a batch that has no mixed question."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HPARAMS["weight_decay"])
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(labels.astype(np.float32))),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
        generator=generator,
    )
    steps = 0
    skipped = 0
    all_positive_batches = 0
    mixed_groups = 0
    pairs = 0
    for _epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scores = model(batch_x)
            loss, info = energy_batch_loss(scores, batch_y, arm)
            mixed_groups += int(info["mixed_questions"])
            pairs += int(info["pairs"])
            all_positive_batches += int(info["all_positive_batch"])
            if loss is None:
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            optimizer.step()
            steps += 1
    return {
        "optimizer_steps": int(steps),
        "skipped_batches": int(skipped),
        "all_positive_batches": int(all_positive_batches),
        "mixed_groups": int(mixed_groups),
        "pairs": int(pairs),
    }


def fit_energy_model(
    x_warm: np.ndarray,
    y_warm: np.ndarray,
    x_fine: np.ndarray,
    y_fine: np.ndarray,
    seeds: Dict[str, int],
    device: torch.device,
    arm: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Warmup on the supplied rows, then finetune. Initialization matches B12."""
    if arm not in SELECTABLE_ARMS:
        raise ValueError(f"unknown energy arm {arm}")
    np.random.seed(int(seeds["warmup"]))
    torch.manual_seed(int(seeds["warmup"]))
    model = CrossCandidateAttentionModule(
        input_dim=int(x_warm.shape[-1]),
        hidden_dim=HPARAMS["hidden_dim"],
        num_layers=HPARAMS["num_layers"],
        num_heads=HPARAMS["num_heads"],
        dropout=HPARAMS["dropout"],
        eps=HPARAMS["layernorm_eps"],
    ).to(device)
    started = time.perf_counter()
    warm = _run_phase(
        model, x_warm, y_warm, HPARAMS["warmup_epochs"], HPARAMS["lr_warmup"],
        int(seeds["warmup"]), device, arm,
    )
    fine = _run_phase(
        model, x_fine, y_fine, HPARAMS["finetune_epochs"], HPARAMS["lr_finetune"],
        int(seeds["finetune"]), device, arm,
    )
    model.eval()
    return state_dict_to_numpy(model), {
        "warmup": warm,
        "finetune": fine,
        "fit_groups": {"warmup": label_group_counts(y_warm), "finetune": label_group_counts(y_fine)},
        "fit_seconds": float(time.perf_counter() - started),
    }


def _cluster_probabilities(
    scores: np.ndarray,
    table: Dict[str, np.ndarray],
    rows: np.ndarray,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax over cluster energies at rho = 0. Shape [R, 4], unused slots are 0."""
    rows = np.asarray(rows, dtype=np.int64)
    used = np.asarray(scores, dtype=np.float64)[rows]
    index = table["cluster_index"][rows]
    counts = table["cluster_count"][rows]
    member_ok = index >= 0
    safe_index = np.clip(index, 0, 3)
    gathered = used[np.arange(rows.size)[:, None, None], safe_index]
    scaled = np.where(member_ok, gathered / float(temperature), -1e30)
    peak = np.max(scaled, axis=2)
    member_total = np.sum(np.exp(np.clip(scaled - peak[:, :, None], -80.0, 80.0)) * member_ok, axis=2)
    log_sum = np.where(member_ok.any(axis=2), peak + np.log(np.maximum(member_total, 1e-30)), -1e30)
    cluster_ok = counts > 0
    peak_energy = np.max(np.where(cluster_ok, log_sum, -1e30), axis=1, keepdims=True)
    weights = np.exp(np.clip(log_sum - peak_energy, -80.0, 80.0)) * cluster_ok
    total = np.sum(weights, axis=1, keepdims=True)
    return weights / np.maximum(total, 1e-30), table["gold_cluster"][rows]


def _brier(probabilities: np.ndarray, gold: np.ndarray, cluster_ok: np.ndarray) -> float:
    indicator = np.zeros_like(probabilities)
    has_gold = gold >= 0
    indicator[np.arange(gold.size), np.clip(gold, 0, 3)] = has_gold
    residual = np.where(cluster_ok, probabilities - indicator, 0.0)
    return float(np.mean(np.sum(residual * residual, axis=1)))


def _ece(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.size == 0:
        return None
    bins = np.minimum((confidence * 10.0).astype(np.int64), 9)
    total = float(confidence.size)
    score = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        score += (count / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(score)


def standalone_report(
    scores: np.ndarray,
    labels: np.ndarray,
    hard: np.ndarray,
    bank: Sequence[Dict[str, Any]],
    parses,
    eligible: np.ndarray,
    table: Dict[str, np.ndarray],
    rows: np.ndarray,
    temperature: float,
) -> Dict[str, Any]:
    """One raw view, cluster decoder at rho = 0. Temperature is not a fusion weight."""
    rows = np.asarray(rows, dtype=np.int64)
    scored = score_aligned(
        [np.asarray(scores, dtype=np.float64)],
        labels, hard, bank, parses, eligible, table,
        [1.0], float(temperature), 0.0, rows,
    )
    choices = np.asarray(scored.pop("choices"), dtype=np.int64)
    probabilities, gold = _cluster_probabilities(scores, table, rows, temperature)
    cluster_ok = table["cluster_count"][rows] > 0
    chosen_cluster = np.full(rows.size, -1, dtype=np.int64)
    confidence = np.zeros(rows.size, dtype=np.float64)
    index = table["cluster_index"][rows]
    for local, choice in enumerate(choices):
        hits = np.any(index[local] == int(choice), axis=1)
        if bool(hits.any()):
            cluster_id = int(np.flatnonzero(hits)[0])
            chosen_cluster[local] = cluster_id
            confidence[local] = float(probabilities[local, cluster_id])
    correct = labels[rows, choices] >= 0.5
    hard_rows = np.asarray(hard, dtype=bool)[rows]
    return {
        "temperature": float(temperature),
        "hard_correct": int(scored["hard_correct"]),
        "hard_total": int(scored["hard_total"]),
        "all_correct": int(scored["all_correct"]),
        "all_total": int(scored["all_total"]),
        "nll_mean": float(scored["nll_sum"] / max(1, int(scored["nll_count"]))),
        "nll_sum": float(scored["nll_sum"]),
        "brier_all": _brier(probabilities, gold, cluster_ok),
        "brier_hard": _brier(probabilities[hard_rows], gold[hard_rows], cluster_ok[hard_rows]),
        "ece_all": _ece(confidence, correct.astype(np.float64)),
        "ece_hard": _ece(confidence[hard_rows], correct[hard_rows].astype(np.float64)),
        "absent_gold_clusters": int(scored["absent_gold_clusters"]),
    }


def select_diagnostic_temperature(reports: Sequence[Dict[str, Any]]) -> int:
    """Lowest mean cluster NLL. Equal NLL keeps the earlier grid temperature."""
    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, int]:
        index, report = item
        return (float(report["nll_mean"]), index)

    index, _report = min(enumerate(reports), key=sort_key)
    return int(index)
