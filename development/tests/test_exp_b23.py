"""EXP-B23 weight rule, loss reduction, and the audit's validation lock."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.exp_b07_cross_attn import run_phase
from experiments.exp_b23_hard_weighting import (
    WEIGHT_CORRECT_NOT_HARD,
    WEIGHT_INCORRECT,
    WEIGHT_OTHER,
    complement_rows,
    internal_gain_percentage_points,
    internal_gate_passed,
    question_loss_weights,
)
from models.cross_attn_net import CrossCandidateAttentionModule, loss_components

AUDIT_PATH = Path(__file__).resolve().parents[1] / "runs" / "audit_logs" / "exp_b23_hard_weighting_audit.json"
CONTRACT_PATH = AUDIT_PATH.parent / "private_configuration_omitted.json"
SOURCE_PATH = DEV_SRC / "experiments" / "exp_b23_hard_weighting.py"


def test_loss_weights_follow_the_three_buckets():
    correct = np.array([False, False, True, True])
    hard = np.array([True, False, False, True])
    weights = question_loss_weights(correct, hard)
    assert weights.tolist() == [WEIGHT_INCORRECT, WEIGHT_INCORRECT, WEIGHT_CORRECT_NOT_HARD, WEIGHT_OTHER]
    assert weights.tolist() == [2.0, 2.0, 0.5, 1.0]


def test_complement_rows_exclude_the_held_out_folds():
    folds = [np.arange(i * 2, i * 2 + 2) for i in range(5)]
    rows = complement_rows(folds, [0, 2])
    assert set(rows.tolist()) == {2, 3, 6, 7, 8, 9}
    assert set(rows.tolist()).isdisjoint({0, 1, 4, 5})


def test_one_percentage_point_is_the_internal_threshold():
    old = [(50, 100), (50, 100)]
    short = [(50, 100), (51, 100)]
    enough = [(50, 100), (52, 100)]
    assert internal_gain_percentage_points(old, short) == 0.5
    assert internal_gain_percentage_points(old, enough) == 1.0
    assert internal_gate_passed(0.999, 1.0) is False
    assert internal_gate_passed(1.0, 1.0) is True
    assert internal_gate_passed(float("nan"), 1.0) is False


def test_contract_weights_stay_at_the_preregistered_values():
    spec = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["followup_batches"]["EXP-B23"]
    weights = spec["sampling_loss_weights"]
    assert weights["training_prediction_incorrect"] == 2.0
    assert weights["training_prediction_correct_and_not_hard"] == 0.5
    assert weights["other"] == 1.0
    assert spec["train_internal_min_heldout_hard_gain_percentage_points"] == 1.0


def test_uniform_weights_match_the_unweighted_mean_and_steps_are_counted():
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = CrossCandidateAttentionModule(
        input_dim=4, hidden_dim=8, num_layers=1, num_heads=2, dropout=0.0,
    ).to(device)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(6, 4, 4)).astype(np.float32)
    labels = np.zeros((6, 4), dtype=np.float32)
    labels[:, 0] = 1.0
    question_weight = np.array([2.0, 2.0, 0.5, 0.5, 1.0, 1.0], dtype=np.float32)
    with torch.no_grad():
        scores = model(torch.from_numpy(features))
        listwise, bce = loss_components(scores, torch.from_numpy(labels))
        per_row = listwise + bce
        uniform = torch.ones(per_row.shape[0])
        uniform_loss = (uniform * per_row).sum() / uniform.sum()
        weighted_loss = (torch.from_numpy(question_weight) * per_row).sum() / question_weight.sum()
    assert torch.allclose(uniform_loss, per_row.mean())
    assert abs(float(weighted_loss) - float(per_row.mean())) > 1e-6
    stats = run_phase(
        model, features, labels, 1, 0.0, 7, device, 1.0, sample_weight=question_weight,
    )
    assert stats["optimizer_steps"] == 1
    assert abs(stats["weight_sum"] - float(question_weight.sum())) < 1e-5
    assert abs(stats["loss"] - float(weighted_loss)) < 1e-5


def test_validation_split_is_constructed_only_inside_confirmation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    marker = 'SplitData("validation")'
    assert source.count(marker) == 1
    assert source.index(marker) > source.index("def run_validation_confirmation")
    assert 'SplitData("test")' not in source
    assert "model_bundle" not in source.replace("written_to_model_bundle", "")


def test_exp_b23_audit_records_whether_validation_was_opened():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    gate = audit["internal_gate"]
    assert gate["minimum_percentage_points"] == 1.0
    assert len(gate["weighted_folds"]) == 5
    assert len(gate["baseline_folds"]) == 5
    assert audit["test_references_opened"] is False
    assert audit["written_to_model_bundle"] is False
    assert audit["candidate_submission_updated"] is False
    assert gate["validation_opened"] is False
    if audit["verdict"] == "STOPPED_INTERNAL_GATE":
        assert gate["passed"] is False
        assert audit["validation_queries"] == 0
        assert audit["validation_confirmation"] is None
    else:
        assert gate["passed"] is True
        assert audit["validation_queries"] > 0
        assert audit["validation_confirmation"]["historical_reproduction"]["val_hard"] == 300
        assert audit["validation_confirmation"]["historical_reproduction"]["val_all"] == 743
