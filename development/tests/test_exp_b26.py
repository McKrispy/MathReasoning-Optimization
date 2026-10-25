"""EXP-B26 centroid weights, C0/C1 scores, and the validation lock."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from models.centroid_verifier import (
    ARMS,
    InvalidCentroidFit,
    fit_centroids,
    linear_c0_expansion,
    question_class_weights,
    score_centroids,
)

AUDIT_PATH = Path(__file__).resolve().parents[1] / "runs" / "audit_logs" / "exp_b26_centroid_audit.json"
SOURCE_PATH = DEV_SRC / "experiments" / "exp_b26_centroid_verifier.py"


def _labels(*rows):
    return np.asarray(rows, dtype=np.float64)


def _parses(*rows):
    return [list(row) for row in rows]


def test_question_class_weights_share_one_unit_and_skip_unparsed():
    labels = _labels([1, 1, 1, 0], [1, 0, 0, 0])
    parses = _parses(["1", "1", "1", "0"], ["1", None, "0", "0"])
    hard = np.array([True, True])
    packed = question_class_weights(labels, parses, hard, np.arange(2))
    assert packed["weights"][1] == [1 / 3, 1 / 3, 1 / 3, 1.0]
    assert packed["weights"][0] == [1.0, 0.5, 0.5]
    assert packed["questions"] == {0: 2, 1: 2}
    assert packed["skipped_unparsed"][0] == 1
    assert packed["points"][0][0] == (0, 3)


def test_easy_rows_do_not_move_the_centroid():
    embeddings = np.zeros((2, 4, 2), dtype=np.float64)
    embeddings[0, 0] = [1.0, 0.0]
    embeddings[0, 1] = [0.0, 1.0]
    embeddings[1, 0] = [50.0, 50.0]
    embeddings[1, 1] = [-50.0, -50.0]
    labels = _labels([1, 0, 0, 0], [1, 0, 0, 0])
    parses = _parses(["a", "b", None, None], ["a", "b", None, None])
    hard = np.array([True, False])
    model = fit_centroids("C0", embeddings, labels, parses, hard, np.arange(2))
    assert model["class_counts"]["questions"] == {"wrong": 1, "correct": 1}
    np.testing.assert_allclose(model["correct"][0], [1.0, 0.0])
    np.testing.assert_allclose(model["wrong"][0], [0.0, 1.0])
    scored = score_centroids(model, embeddings)
    assert scored.shape == (2, 4)


def test_c0_matches_the_expanded_linear_form():
    rng = np.random.default_rng(0)
    correct = rng.normal(size=8)
    wrong = rng.normal(size=8)
    vectors = rng.normal(size=(5, 8))
    model = {
        "arm": "C0",
        "correct": correct[None, :],
        "wrong": wrong[None, :],
        "tau": None,
    }
    block = np.repeat(vectors[:, None, :], 4, axis=1)
    direct = score_centroids(model, block)[:, 0]
    expanded = linear_c0_expansion(vectors, correct, wrong)
    np.testing.assert_allclose(direct, expanded, atol=1e-8)


def test_missing_class_stops():
    embeddings = np.zeros((1, 4, 2))
    embeddings[0, 0] = [1.0, 0.0]
    labels = _labels([1, 1, 0, 0])
    parses = _parses(["a", "a", None, None])
    with pytest.raises(InvalidCentroidFit):
        fit_centroids("C0", embeddings, labels, parses, np.array([True]), np.array([0]))


def test_c1_caps_k_and_floors_tau():
    embeddings = np.zeros((1, 4, 2), dtype=np.float64)
    embeddings[0, 0] = [0.0, 0.0]
    embeddings[0, 1] = [0.0, 0.0]
    embeddings[0, 2] = [1.0, 0.0]
    embeddings[0, 3] = [0.0, 1.0]
    labels = _labels([1, 1, 0, 0])
    parses = _parses(["a", "a", "b", "c"])
    model = fit_centroids("C1", embeddings, labels, parses, np.array([True]), np.array([0]))
    assert model["requested_k"]["correct"] == 1
    assert model["requested_k"]["wrong"] == 2
    assert model["actual_k"]["correct"] == 1
    assert model["actual_k"]["wrong"] == 2
    assert model["tau"] >= 1e-6
    identical = np.zeros((1, 4, 2))
    identical[0, 0] = [1.0, 0.0]
    identical[0, 1] = [-1.0, 0.0]
    flat = fit_centroids("C1", identical, labels, parses, np.array([True]), np.array([0]))
    assert flat["tau"] == 1e-6
    assert ARMS == ("C0", "C1")


def test_c1_logmeanexp_uses_a_uniform_prototype_prior():
    correct = np.array([[0.0, 0.0], [1.0, 0.0]])
    wrong = np.array([[0.0, 2.0]])
    model = {"arm": "C1", "correct": correct, "wrong": wrong, "tau": 1.0}
    point = np.zeros((1, 4, 2))
    point[0, 0] = [0.0, 0.0]
    score = float(score_centroids(model, point)[0, 0])
    correct_term = np.log(0.5 * (np.exp(0.0) + np.exp(-1.0)))
    wrong_term = np.log(np.exp(-4.0))
    assert abs(score - (correct_term - wrong_term)) < 1e-8


def test_validation_split_is_constructed_only_inside_confirmation():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    marker = 'SplitData("validation")'
    assert source.count(marker) == 1
    assert source.index(marker) > source.index("def run_validation_confirmation")
    assert 'SplitData("test")' not in source
    assert "model_bundle" not in source.replace("written_to_model_bundle", "")


def test_exp_b26_audit_records_whether_validation_was_opened():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    gate = audit["internal_gate"]
    assert gate["minimum_mean_fold_hard_gain_percentage_points"] == 0.5
    assert len(audit["outer_folds"]) == 5
    assert audit["test_references_opened"] is False
    assert audit["written_to_model_bundle"] is False
    assert audit["candidate_submission_updated"] is False
    assert audit["reference_system"] == "B25 fold-local locked three-view decoder"
    if audit["verdict"] == "STOPPED_INTERNAL_GATE":
        assert gate["passed"] is False
        assert audit["validation_queries"] == 0
        assert audit["validation_confirmation"] is None
    elif audit["verdict"] == "FAILED":
        raise AssertionError(audit.get("error", "B26 failed"))
    else:
        assert gate["passed"] is True
        assert audit["validation_queries"] == 1
        confirmation = audit["validation_confirmation"]
        assert confirmation["historical_reproduction"]["val_hard"] == 300
        assert confirmation["historical_reproduction"]["val_all"] == 743
        assert confirmation["strict_reproduction"]["val_hard"] == 295
        assert confirmation["strict_reproduction"]["val_all"] == 738
