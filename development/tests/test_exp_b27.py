"""EXP-B27 energy losses, diagnostic temperature, and the validation lock."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softplus

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b07_cross_attn import HPARAMS
from models.energy_verifier import (
    DIAGNOSTIC_TEMPERATURES,
    SELECTABLE_ARMS,
    bradley_terry_loss,
    energy_batch_loss,
    fit_energy_model,
    select_diagnostic_temperature,
    standalone_report,
    summed_bce,
)
from postprocessing.aligned_fusion import build_cluster_table, question_eligibility

AUDIT_PATH = Path(__file__).resolve().parents[1] / "runs" / "audit_logs" / "exp_b27_energy_audit.json"
SOURCE_PATH = DEV_SRC / "experiments" / "exp_b27_energy_verifier.py"


def _explicit_bt(scores, labels):
    question_means = []
    for score, label in zip(scores, labels):
        positive = np.flatnonzero(np.asarray(label) >= 0.5)
        negative = np.flatnonzero(np.asarray(label) < 0.5)
        if positive.size == 0 or negative.size == 0:
            continue
        gaps = [
            float(softplus(score[neg] - score[pos]))
            for pos in positive
            for neg in negative
        ]
        question_means.append(float(np.mean(gaps)))
    return float(np.mean(question_means))


def test_bt_matches_the_per_question_pair_mean():
    scores = torch.tensor(
        [
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    loss, info = bradley_terry_loss(scores, labels)
    expected = _explicit_bt(scores.numpy(), labels.numpy())
    assert abs(float(loss) - expected) < 1e-6
    assert info["mixed_questions"] == 2
    assert info["pairs"] == 3 + 4
    pooled = []
    for score, label in zip(scores.numpy(), labels.numpy()):
        positive = np.flatnonzero(label >= 0.5)
        negative = np.flatnonzero(label < 0.5)
        pooled.extend(float(softplus(score[neg] - score[pos])) for pos in positive for neg in negative)
    assert abs(float(loss) - float(np.mean(pooled))) > 1e-3


def test_all_positive_batch_has_no_bt_loss():
    scores = torch.zeros((2, 4), dtype=torch.float64)
    labels = torch.ones((2, 4), dtype=torch.float64)
    loss, info = bradley_terry_loss(scores, labels)
    assert loss is None
    assert info["mixed_questions"] == 0
    assert info["pairs"] == 0


def test_e2_adds_the_summed_bce_and_e1_does_not():
    scores = torch.tensor([[0.2, -0.4, 0.1, 0.0]], dtype=torch.float64)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    e1, info = energy_batch_loss(scores, labels, "E1")
    e2, _info = energy_batch_loss(scores, labels, "E2")
    bt, _bt_info = bradley_terry_loss(scores, labels)
    bce = summed_bce(scores, labels)
    assert abs(float(e1) - float(bt)) < 1e-8
    assert abs(float(e2) - (float(bt) + float(bce))) < 1e-8
    assert info["all_positive_batch"] == 0
    assert SELECTABLE_ARMS == ("E1", "E2")
    assert DIAGNOSTIC_TEMPERATURES == (0.25, 0.5, 1.0, 2.0)


def test_e1_skips_updates_when_the_batch_has_no_pairs():
    features = np.zeros((4, 4, 8), dtype=np.float32)
    labels = np.ones((4, 4), dtype=np.float32)
    _state, stats = fit_energy_model(
        features, labels, features, labels,
        {"warmup": 7, "finetune": 8}, torch.device("cpu"), "E1",
    )
    assert stats["warmup"]["optimizer_steps"] == 0
    assert stats["warmup"]["skipped_batches"] == HPARAMS["warmup_epochs"]
    assert stats["finetune"]["optimizer_steps"] == 0
    _state, trained = fit_energy_model(
        features, labels, features, labels,
        {"warmup": 7, "finetune": 8}, torch.device("cpu"), "E2",
    )
    assert trained["warmup"]["optimizer_steps"] == HPARAMS["warmup_epochs"]


def test_diagnostic_temperature_uses_cluster_nll_and_keeps_the_earlier_tie():
    bank = [{
        "id": "q0",
        "candidates": [
            {"text": "#### 3"},
            {"text": "#### 3"},
            {"text": "#### 4"},
            {"text": "#### 4"},
        ],
    }]
    labels = np.array([[1.0, 1.0, 0.0, 0.0]])
    hard = np.array([True])
    scores = np.array([[5.0, 5.0, 0.0, 0.0]])
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    report = standalone_report(
        scores, labels, hard, bank, parses, eligible, table, np.array([0]), 1.0,
    )
    probability = np.exp(5.0) / (np.exp(5.0) + 1.0)
    assert abs(report["nll_mean"] - (-np.log(probability))) < 1e-6
    assert abs(report["brier_all"] - 2.0 * (1.0 - probability) ** 2) < 1e-6
    assert abs(report["ece_all"] - abs(1.0 - probability)) < 1e-6
    assert report["hard_correct"] == 1
    reports = []
    for temperature in DIAGNOSTIC_TEMPERATURES:
        reports.append(standalone_report(
            scores, labels, hard, bank, parses, eligible, table, np.array([0]), temperature,
        ))
    assert select_diagnostic_temperature(reports) == 0
    tied = [{"nll_mean": 0.4}, {"nll_mean": 0.4}, {"nll_mean": 0.5}]
    assert select_diagnostic_temperature(tied) == 0


def test_validation_split_is_constructed_only_inside_confirmation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    marker = 'SplitData("validation")'
    assert source.count(marker) == 1
    assert source.index(marker) > source.index("def run_validation_confirmation")
    assert 'SplitData("test")' not in source
    assert "model_bundle" not in source.replace("written_to_model_bundle", "")


def test_exp_b27_audit_records_whether_validation_was_opened():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    gate = audit["internal_gate"]
    assert gate["minimum_mean_fold_hard_gain_percentage_points"] == 0.5
    assert len(audit["outer_folds"]) == 5
    assert audit["test_references_opened"] is False
    assert audit["written_to_model_bundle"] is False
    assert audit["candidate_submission_updated"] is False
    assert audit["reference_system"] == "B25 fold-local locked three-view decoder"
    assert audit["e0_reused_from_b25"] is True
    if audit["verdict"] == "STOPPED_INTERNAL_GATE":
        assert gate["passed"] is False
        assert audit["validation_queries"] == 0
        assert audit["validation_confirmation"] is None
    elif audit["verdict"] == "FAILED":
        raise AssertionError(audit.get("error", "B27 failed"))
    else:
        assert gate["passed"] is True
        assert audit["validation_queries"] == 1
        confirmation = audit["validation_confirmation"]
        assert confirmation["historical_reproduction"]["val_hard"] == 300
        assert confirmation["historical_reproduction"]["val_all"] == 743
        assert confirmation["strict_reproduction"]["val_hard"] == 295
        assert confirmation["strict_reproduction"]["val_all"] == 738
