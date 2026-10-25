"""EXP-B25 decoder alignment: selection rule, scale, gate, and the validation lock."""
import json
import sys
from pathlib import Path

import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from experiments.multiseed import SEEDS, fold_model_seeds, full_model_seeds
from postprocessing.aligned_fusion import (
    build_cluster_table,
    choose_from_scores,
    cluster_log_probabilities,
    fit_aligned_weights,
    fit_global_rms,
    internal_gate_decision,
    nested_attention_seeds,
    question_eligibility,
)
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
)

AUDIT_PATH = Path(__file__).resolve().parents[1] / "runs" / "audit_logs" / "exp_b25_decoder_alignment_audit.json"
CONTRACT_PATH = AUDIT_PATH.parent / "private_configuration_omitted.json"
SOURCE_PATH = DEV_SRC / "experiments" / "exp_b25_decoder_alignment.py"


def _bank(texts):
    return [{"candidates": [{"text": text} for text in row]} for row in texts]


def test_rho_zero_matches_the_deployed_aggregator():
    texts = [
        ["#### 1", "#### 2", "#### 1", "#### 3"],
        ["no answer", "#### 4", "#### 5", "#### 4"],
        ["step", "without", "a final", "answer"],
        ["#### **Job 3**\n\\boxed{250}", "#### 9", "#### 9", "#### 8"],
    ]
    bank = _bank(texts)
    scores = np.array([
        [2.0, 0.2, 1.4, -0.4],
        [3.0, 0.1, 0.2, 1.7],
        [0.2, 1.5, -0.3, 0.4],
        [4.0, 0.5, 0.4, 0.2],
    ])
    eligible, parses = question_eligibility(bank)
    chosen = choose_from_scores(scores, bank, parses, eligible, temperature=0.5, rho=0.0)
    masked = apply_conflict_penalty(apply_validity_mask(scores, bank, penalty=1e9), bank, penalty=1e9)
    expected = SoftmaxClusterAggregator(temperature=0.5).aggregate(bank, masked)
    assert chosen.tolist() == expected


def test_positive_rho_downweights_a_repeated_answer():
    bank = _bank([["#### 1", "#### 2", "#### 1", "#### 3"]])
    scores = np.array([[2.0, 2.05, 2.0, 0.0]])
    eligible, parses = question_eligibility(bank)
    without = choose_from_scores(scores, bank, parses, eligible, temperature=0.5, rho=0.0)
    with_rho = choose_from_scores(scores, bank, parses, eligible, temperature=0.5, rho=1.0)
    assert int(without[0]) == 0
    assert int(with_rho[0]) == 1


def test_global_scale_ignores_ineligible_scores_and_sentinels():
    scores = np.array([[0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 0.0, -1e9]])
    eligible = np.array([[True, True, True, False], [True, True, True, True]])
    rows = np.array([0])
    rms = fit_global_rms(scores, eligible, rows)
    assert abs(rms - 25.0) < 1e-9
    sentinel_rms = fit_global_rms(scores, eligible, np.array([1]))
    assert sentinel_rms == 1e-6
    flat = fit_global_rms(np.zeros((1, 4)), np.ones((1, 4), dtype=bool), np.array([0]))
    assert flat == 1e-6


def test_missing_gold_cluster_is_clipped_and_kept():
    bank = _bank([["nope", "#### 2", "#### 3", "#### 4"]])
    labels = np.array([[1.0, 0.0, 0.0, 0.0]])
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    log_prob, has_gold = cluster_log_probabilities(
        np.zeros((1, 4)), table, np.array([0]), temperature=0.5, rho=0.0,
    )
    assert bool(has_gold[0]) is False
    assert abs(float(log_prob[0]) - np.log(1e-12)) < 1e-9


def test_contract_has_seven_configurations_and_no_cross():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    from postprocessing.aligned_fusion import b25_configs

    configs = b25_configs(contract)
    assert len(configs) == 7
    assert [item["arm"] for item in configs] == ["D1", "D2", "D2", "D2", "D2", "D3", "D3"]
    assert [item["temperature"] for item in configs if item["arm"] == "D2"] == [0.25, 0.5, 1.0, 2.0]
    assert all(item["rho"] == 0.0 for item in configs if item["arm"] == "D2")
    assert [item["rho"] for item in configs if item["arm"] == "D3"] == [0.5, 1.0]
    assert all(item["normalization"] == "question_zscore" for item in configs if item["arm"] == "D3")


def test_nested_seeds_are_unique_and_disjoint_from_fold_seeds():
    occupied = set()
    for seed in SEEDS:
        occupied.update(full_model_seeds(seed).values())
        for fold_id in range(5):
            occupied.update(fold_model_seeds(seed, fold_id).values())
    nested = []
    for variant in (0, 1):
        for seed in SEEDS:
            for outer in range(5):
                for inner in range(5):
                    nested.extend(nested_attention_seeds(variant, seed, outer, inner).values())
    assert len(nested) == len(set(nested)) == 300
    assert occupied.isdisjoint(nested)
    assert min(nested) >= 1_000_000


def test_internal_gate_requires_both_rates_three_folds_and_all_count():
    def row(hard, total, all_correct):
        return {
            "hard_correct": hard,
            "hard_total": total,
            "all_correct": all_correct,
            "all_total": total,
        }

    reference = [row(5, 10, 5), row(5, 10, 5), row(5, 10, 5), row(5, 10, 5), row(480, 960, 480)]
    mean_only = [row(6, 10, 6), row(6, 10, 6), row(6, 10, 6), row(5, 10, 5), row(480, 960, 480)]
    decision = internal_gate_decision(reference, mean_only, 0.5, 3)
    assert decision["mean_hard_gain_percentage_points"] > 0.5
    assert decision["pooled_hard_gain_percentage_points"] < 0.5
    assert decision["positive_outer_folds"] == 3
    assert decision["passed"] is False

    both = [row(6, 10, 6), row(6, 10, 6), row(6, 10, 6), row(6, 10, 6), row(490, 960, 490)]
    passed = internal_gate_decision(reference, both, 0.5, 3)
    assert passed["pooled_all_delta"] >= 0
    assert passed["passed"] is True

    all_down = [row(6, 10, 1), row(6, 10, 1), row(6, 10, 1), row(6, 10, 1), row(490, 960, 470)]
    dropped = internal_gate_decision(reference, all_down, 0.5, 3)
    assert dropped["pooled_all_delta"] < 0
    assert dropped["passed"] is False


def test_fit_returns_a_successful_simplex():
    scores = np.zeros((6, 4), dtype=np.float64)
    scores[:, 0] = 2.0
    views = [scores, np.zeros((6, 4)), np.zeros((6, 4))]
    labels = np.zeros((6, 4))
    labels[:, 0] = 1.0
    bank = _bank([["#### 1", "#### 2", "#### 3", "#### 4"] for _ in range(6)])
    eligible, parses = question_eligibility(bank)
    table = build_cluster_table(parses, eligible, labels)
    weights, info = fit_aligned_weights(
        views, np.ones(6, dtype=bool), table, 0.5, 0.0, np.arange(6),
    )
    assert abs(float(weights.sum()) - 1.0) < 1e-8
    assert float(weights.min()) >= -1e-10
    assert info["absent_gold_clusters"] == 0
    assert weights[0] > weights[1]


def test_validation_split_is_constructed_only_inside_confirmation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    marker = 'SplitData("validation")'
    assert source.count(marker) == 1
    assert source.index(marker) > source.index("def run_validation_confirmation")
    assert 'SplitData("test")' not in source
    assert "model_bundle" not in source.replace("written_to_model_bundle", "")


def test_exp_b25_audit_records_whether_validation_was_opened():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    gate = audit["internal_gate"]
    assert gate["minimum_mean_fold_hard_gain_percentage_points"] == 0.5
    assert gate["minimum_pooled_hard_gain_percentage_points"] == 0.5
    assert len(audit["outer_folds"]) == 5
    assert audit["training"]["nested_attention_models"] == 120
    assert audit["training"]["new_base_training_recipe"] is False
    assert audit["test_references_opened"] is False
    assert audit["written_to_model_bundle"] is False
    assert audit["candidate_submission_updated"] is False
    assert len(audit["configurations"]) == 7
    if audit["verdict"] == "STOPPED_INTERNAL_GATE":
        assert gate["passed"] is False
        assert gate["validation_opened"] is False
        assert audit["validation_queries"] == 0
        assert audit["validation_confirmation"] is None
        assert audit["validation_opened"] is False
    elif audit["verdict"] == "FAILED":
        raise AssertionError(audit.get("error", "B25 failed"))
    else:
        assert gate["passed"] is True
        assert audit["validation_queries"] == 1
        assert audit["validation_confirmation"]["historical_reproduction"]["val_hard"] == 300
        assert audit["validation_confirmation"]["historical_reproduction"]["val_all"] == 743
        assert audit["validation_confirmation"]["strict_reproduction"]["val_hard"] == 295
        assert audit["validation_confirmation"]["strict_reproduction"]["val_all"] == 738
