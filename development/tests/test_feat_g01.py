"""Test FEAT-G01: ConsensusFeatureExtractor correctness and performance."""
from pathlib import Path
import sys
import time
import numpy as np

DEV_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = DEV_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
OFFICIAL_CODE_DIR = DEV_ROOT.parent / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

from common_utils import load_dataset
from features.consensus import ConsensusFeatureExtractor


def test_feat_g01_unit():
    extractor = ConsensusFeatureExtractor()
    assert extractor.feature_dim == 11

    # Synthetic test question
    mock_bank = [
        {
            "id": "mock_01",
            "question": "Janet has 10 apples and buys 5 more. How many?",
            "candidates": [
                {"index": 0, "text": "She has 10 + 5 = 15 apples. #### 15"},
                {"index": 1, "text": "She has 15 apples. #### 15"},
                {"index": 2, "text": "She has 10 apples. #### 10"},
                {"index": 3, "text": "Cannot determine."},  # unparsed
            ]
        }
    ]
    mock_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}

    feats = extractor.transform(mock_bank, mock_arrays)
    assert feats.shape == (1, 4, 11)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()

    # Cand 0: valid (1), count=2, ratio=2/3=0.6667, unique_maj=1, tied_maj=0, distinct=2, log(16), int=1, neg=0, zero=0, appears=0
    c0 = feats[0, 0]
    assert c0[0] == 1.0  # is_valid
    assert c0[1] == 2.0  # cluster_count
    assert abs(c0[2] - 2/3) < 1e-4  # cluster_ratio
    assert c0[3] == 1.0  # is_unique_majority
    assert c0[4] == 0.0  # is_tied_majority
    assert c0[5] == 2.0  # num_distinct_answers
    assert c0[7] == 1.0  # is_integer
    assert c0[8] == 0.0  # is_negative
    assert c0[9] == 0.0  # is_zero
    assert c0[10] == 0.0 # 15 not in question (10 and 5 are)

    # Cand 2: valid (1), count=1, ratio=1/3, unique_maj=0, appears_in_question=1 (10 is in question)
    c2 = feats[0, 2]
    assert c2[0] == 1.0
    assert c2[1] == 1.0
    assert c2[3] == 0.0
    assert c2[10] == 1.0  # 10 appears in question!

    # Cand 3: invalid parse
    c3 = feats[0, 3]
    assert c3[0] == 0.0
    assert c3[1] == 0.0
    assert c3[5] == 2.0

    print("FEAT-G01 Synthetic Unit Tests PASSED!")


def test_feat_g01_dataset():
    extractor = ConsensusFeatureExtractor()
    print("Testing FEAT-G01 on validation set (798 samples)...")
    val_bank, val_arrays = load_dataset('validation')

    t0 = time.perf_counter()
    feats = extractor.transform(val_bank, val_arrays)
    elapsed = time.perf_counter() - t0

    assert feats.shape == (798, 4, 11)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Found non-finite values in FEAT-G01!"

    print(f"Validation extraction complete in {elapsed:.3f} s ({(elapsed/798)*1000:.2f} ms/question).")
    print(f"Feature names: {extractor.FEATURE_NAMES}")
    print(f"Feature means across candidates:\n" + "\n".join(
        f"  {extractor.FEATURE_NAMES[k]:22s}: {feats[:, :, k].mean():.4f}" for k in range(11)
    ))
    print("FEAT-G01 Dataset Test PASSED!")


if __name__ == "__main__":
    test_feat_g01_unit()
    test_feat_g01_dataset()
