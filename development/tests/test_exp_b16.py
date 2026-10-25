"""Unit tests for EXP-B16 postprocessing and de-mining masking (POST-P05 & POST-P06)."""
from pathlib import Path
import sys

import numpy as np
import pytest

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / "DASE7506-MP2" / "code"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
    detect_parser_heading_conflict,
)


def test_detect_parser_heading_conflict():
    # Normal reasoning ending with ####
    normal_1 = "First we do 2 + 2 = 4. \n#### 4"
    assert not detect_parser_heading_conflict(normal_1)

    # Normal reasoning ending with \\boxed
    normal_2 = "First we do 2 + 2 = 4. The answer is \\boxed{4}"
    assert not detect_parser_heading_conflict(normal_2)

    # Markdown heading trap: body contains #### **Job 3**, but final answer is in \\boxed{250}
    trap_1 = (
        "Let us calculate the cost.\n"
        "#### **Job 3**\n"
        "For job 3, the cost is 250 dollars.\n"
        "Therefore, the total answer is \\boxed{250}."
    )
    # common.parse_answer will split on #### and see '3', while boxed has '250'
    assert common.parse_answer(trap_1) == "3"
    assert detect_parser_heading_conflict(trap_1)

    # Unparseable text: no numbers
    unparseable = "I do not know the answer."
    assert not detect_parser_heading_conflict(unparseable)


def test_apply_validity_mask():
    bank = [
        {
            "id": "q1",
            "candidates": [
                {"text": "The answer is #### 10"},      # valid
                {"text": "No answer given here."},         # invalid
                {"text": "The answer is #### 20"},      # valid
                {"text": "Another gibberish without num"}, # invalid
            ],
        }
    ]
    logits = np.array([[1.0, 5.0, 2.0, 4.0]], dtype=np.float32)

    # Without mask, argmax is candidate 1 (logit 5.0, which is invalid)
    assert np.argmax(logits, axis=1)[0] == 1

    # With validity mask, candidates 1 and 3 are penalized
    masked = apply_validity_mask(logits, bank, penalty=1e9)
    assert masked[0, 1] <= -1e8
    assert masked[0, 3] <= -1e8
    assert masked[0, 0] == 1.0
    assert masked[0, 2] == 2.0

    # With mask, argmax selects candidate 2 (logit 2.0, valid)
    assert np.argmax(masked, axis=1)[0] == 2


def test_apply_conflict_penalty():
    trap_text = (
        "Let us solve step by step.\n"
        "#### Step 2\n"
        "The calculated total is \\boxed{100}."
    )
    bank = [
        {
            "id": "q1",
            "candidates": [
                {"text": trap_text},                    # heading conflict
                {"text": "The answer is #### 100"},     # safe valid
                {"text": "The answer is #### 50"},      # safe valid
                {"text": "Invalid text"},               # invalid
            ],
        }
    ]
    logits = np.array([[5.0, 4.0, 3.0, 1.0]], dtype=np.float32)
    penalized = apply_conflict_penalty(logits, bank, penalty=1e9)
    assert penalized[0, 0] <= -1e8
    assert penalized[0, 1] == 4.0
    assert np.argmax(penalized, axis=1)[0] == 1


def test_softmax_cluster_aggregator():
    # 2 candidates with answer "42", 1 candidate with answer "100" (high logit), 1 candidate invalid
    bank = [
        {
            "id": "q1",
            "candidates": [
                {"text": "Answer is #### 42"},   # logit 2.0
                {"text": "Answer is #### 42"},   # logit 2.0
                {"text": "Answer is #### 100"},  # logit 2.5
                {"text": "Invalid"},             # logit -1e9
            ],
        }
    ]
    # In raw argmax, candidate 2 ("100") wins with 2.5 vs 2.0
    logits = np.array([[2.0, 2.0, 2.5, -1e9]], dtype=np.float32)

    # In Softmax cluster aggregation with T=1.0:
    # P(c0) ~ exp(2.0), P(c1) ~ exp(2.0). Sum for "42" = 2 * exp(2.0) = 2 * 7.389 = 14.778
    # P(c2) = exp(2.5) = 12.182.
    # Therefore, cluster "42" wins over cluster "100"!
    aggregator = SoftmaxClusterAggregator(temperature=1.0)
    choices = aggregator.aggregate(bank, logits)
    assert choices[0] in [0, 1]  # Either c0 or c1 (both answer 42)
    assert common.parse_answer(bank[0]["candidates"][choices[0]]["text"]) == "42"
