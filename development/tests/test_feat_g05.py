"""Unit tests for FEAT-G05: FineGrainedVerificationFeatureExtractor."""
from pathlib import Path
import sys
import numpy as np
import pytest

DEV_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from features.finegrained import (
    FineGrainedVerificationFeatureExtractor,
    extract_equation_records,
)


def test_extract_equation_records():
    text = (
        "First, John buys 5 apples at $2 each: 5 * 2 = 10.\n"
        "Then he buys 3 oranges at $4 each: 3 * 4 = 12.\n"
        "Total cost is: 10 + 12 = 22.\n"
        "#### 22"
    )
    records = extract_equation_records(text)
    assert len(records) == 3
    assert records[0]["lhs_val"] == 10.0
    assert records[0]["rhs_val"] == 10.0
    assert records[0]["is_correct"] is True
    assert records[0]["operands"] == [5.0, 2.0]

    assert records[1]["lhs_val"] == 12.0
    assert records[1]["rhs_val"] == 12.0
    assert records[1]["is_correct"] is True
    assert records[1]["operands"] == [3.0, 4.0]

    assert records[2]["lhs_val"] == 22.0
    assert records[2]["rhs_val"] == 22.0
    assert records[2]["is_correct"] is True
    assert records[2]["operands"] == [10.0, 12.0]


def test_final_eq_matches_answer():
    ext = FineGrainedVerificationFeatureExtractor()
    bank = [{
        "question": "What is 5 * 2?",
        "candidates": [
            # Candidate 0: Final eq matches parsed answer
            {"text": "We compute 5 * 2 = 10. The answer is #### 10"},
            # Candidate 1: Final eq does NOT match parsed answer (calculation hallucination)
            {"text": "We compute 5 * 2 = 10. The answer is #### 12"},
            # Candidate 2: No equation present
            {"text": "The answer is obviously #### 10"},
            # Candidate 3: No valid parsed answer
            {"text": "We compute 5 * 2 = 10."},
        ]
    }]
    dummy_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}
    feats = ext.transform(bank, dummy_arrays)

    assert feats.shape == (1, 4, 6)
    # Feature 0: final_eq_matches_answer
    assert feats[0, 0, 0] == 1.0
    assert feats[0, 1, 0] == 0.0
    assert feats[0, 2, 0] == 0.0
    assert feats[0, 3, 0] == 0.0


def test_equation_chain_continuity():
    ext = FineGrainedVerificationFeatureExtractor()
    bank = [{
        "question": "There are 10 apples and 5 oranges.",
        "candidates": [
            # Candidate 0: Continuous chain (uses question stem numbers 10, 5, and prior RHS 15)
            {
                "text": (
                    "Step 1: 10 + 5 = 15.\n"
                    "Step 2: 15 - 5 = 10.\n"
                    "#### 10"
                )
            },
            # Candidate 1: Discontinuous chain (step 2 introduces alien numbers 99, 7)
            {
                "text": (
                    "Step 1: 10 + 5 = 15.\n"
                    "Step 2: 99 + 7 = 106.\n"
                    "#### 106"
                )
            },
            # Candidate 2: Single equation (continuity by definition is 1.0)
            {
                "text": "10 + 5 = 15. #### 15"
            },
            # Candidate 3: No equations (continuity by definition is 1.0)
            {
                "text": "#### 15"
            },
        ]
    }]
    dummy_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}
    feats = ext.transform(bank, dummy_arrays)

    # Feature 1: equation_chain_continuity
    assert feats[0, 0, 1] == 1.0
    assert feats[0, 1, 1] < 1.0
    assert feats[0, 2, 1] == 1.0
    assert feats[0, 3, 1] == 1.0


def test_heading_conflict_and_cluster_evidence():
    ext = FineGrainedVerificationFeatureExtractor()
    bank = [{
        "question": "A worker has 3 jobs. Job 1 pays $100, Job 2 pays $150. What is total?",
        "candidates": [
            # Candidate 0: Heading trap (contains '#### **Job 3**' and '\boxed{250}')
            {
                "text": "#### **Job 3**\nTotal is 100 + 150 = 250. Final answer is \\boxed{250}."
            },
            # Candidate 1: Well-verified candidate for answer 250
            {
                "text": "100 + 150 = 250. The answer is #### 250."
            },
            # Candidate 2: Concise candidate for answer 250 without equations (skips steps)
            {
                "text": "Total is 250. #### 250"
            },
            # Candidate 3: Erroneous candidate with wrong equation
            {
                "text": "100 + 150 = 300. #### 300"
            },
        ]
    }]
    dummy_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}
    feats = ext.transform(bank, dummy_arrays)

    # Feature 2: has_parser_heading_conflict
    assert feats[0, 0, 2] == 1.0  # Conflict detected in Candidate 0
    assert feats[0, 1, 2] == 0.0  # Clean in Candidate 1
    assert feats[0, 2, 2] == 0.0  # Clean in Candidate 2
    assert feats[0, 3, 2] == 0.0  # Clean in Candidate 3

    # Candidates 1 and 2 share the same parsed answer '250'
    # Candidate 1 has 1 correct equation, 0 wrong, coverage > 0
    # Candidate 2 has 0 equations
    # Verify cluster collective evidence transferred to Candidate 2:
    # Feature 3: cluster_avg_correct_eq_ratio
    assert feats[0, 1, 3] == feats[0, 2, 3]
    # Feature 4: cluster_max_condition_coverage
    assert feats[0, 2, 4] == feats[0, 1, 4]
    # Feature 5: cluster_min_wrong_eq
    assert feats[0, 2, 5] == 0.0

    # Candidate 3 has 1 wrong equation
    assert feats[0, 3, 5] == 1.0
