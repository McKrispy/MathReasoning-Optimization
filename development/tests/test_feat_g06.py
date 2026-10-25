"""Fixed FEAT-G06 rules. These examples are not derived from validation errors."""
import sys
from pathlib import Path

import numpy as np

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from features.char_hash import (
    DIMENSIONS,
    hash_bucket,
    hashed_counts,
    idf_from_counts,
    normalize_candidate_text,
    tfidf_matrix,
)
from models.linear_view import TextLogisticView, score_exported


def test_normalization_examples_stay_fixed():
    assert normalize_candidate_text("The TOTAL is 60 km/h.") == "the total is 0 km/h."
    assert normalize_candidate_text("Wait  2.50\nmore") == "wait 0 more"
    assert normalize_candidate_text("1,000 apples and 1/4.") == "0 apples and 0/0."
    assert normalize_candidate_text("Answer is 60.") == normalize_candidate_text("Answer is 70.")


def test_hash_is_unsigned_crc32_and_stable():
    assert hash_bucket("cat") == hash_bucket("cat")
    assert 0 <= hash_bucket("cat") < DIMENSIONS
    assert hashed_counts("abcd") == hashed_counts("abcd")


def test_idf_uses_only_the_fit_candidates_and_rows_are_unit_length():
    shared = hashed_counts("alpha beta")
    other = hashed_counts("alpha gamma")
    idf = idf_from_counts([shared, other])
    both = set(shared) & set(other)
    only_one = set(shared) - set(other)
    assert both and only_one
    shared_bucket = next(iter(both))
    single_bucket = next(iter(only_one))
    assert np.isclose(idf[shared_bucket], np.log((1 + 2) / (1 + 2)) + 1)
    assert np.isclose(idf[single_bucket], np.log((1 + 2) / (1 + 1)) + 1)
    assert not np.allclose(idf, idf_from_counts([shared]))
    matrix = tfidf_matrix([shared, other], idf)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_exported_coefficients_match_the_fitted_logistic_scores():
    texts_counts = [hashed_counts(text) for text in ("alpha beta", "alpha gamma", "gamma delta", "beta beta")]
    labels = np.array([1, 0, 1, 0])
    view = TextLogisticView()
    view.fit(texts_counts, texts_counts, labels)
    direct = view.decision_function(texts_counts)
    exported = score_exported(texts_counts, view.export_arrays())
    assert np.allclose(direct, exported, atol=1e-6)
