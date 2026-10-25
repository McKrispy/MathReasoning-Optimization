"""Test FEAT-G03: SymbolicVerificationFeatureExtractor correctness and dataset profiling."""
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
from features.symbolic import SymbolicVerificationFeatureExtractor


def test_feat_g03_unit():
    extractor = SymbolicVerificationFeatureExtractor()
    assert extractor.feature_dim == 14

    mock_bank = [
        {
            "id": "mock_01",
            "question": "Janet has 10 apples and 5 oranges. How many fruits?",
            "candidates": [
                {
                    "index": 0,
                    # All equations correct, covers both 10 and 5
                    "text": "She has 10 apples. Then 10 + 5 = 15 fruits. Total is 15 * 1 = 15. #### 15"
                },
                {
                    "index": 1,
                    # Arithmetic error: 10 + 5 = 20
                    "text": "10 + 5 = 20. Total is 20. #### 20"
                },
                {
                    "index": 2,
                    # Missing number 5 from question entirely
                    "text": "10 * 2 = 20. #### 20"
                },
                {
                    "index": 3,
                    # No equations at all
                    "text": "The answer is clearly 15. #### 15"
                },
            ]
        }
    ]
    mock_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}

    feats = extractor.transform(mock_bank, mock_arrays)
    assert feats.shape == (1, 4, 14)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()

    # Cand 0: 2 correct equations, 0 wrong equations, has_arithmetic_error=0, condition_coverage=1.0 (both 10 and 5)
    c0 = feats[0, 0]
    assert c0[0] == 2.0, f"Expected 2 equations, got {c0[0]}"
    assert c0[1] == 2.0, f"Expected 2 correct equations, got {c0[1]}"
    assert c0[2] == 0.0, f"Expected 0 wrong equations, got {c0[2]}"
    assert c0[4] == 0.0, f"Expected has_arithmetic_error=0, got {c0[4]}"
    assert c0[7] == 1.0, f"Expected coverage=1.0, got {c0[7]}"

    # Cand 1: 1 wrong equation (10 + 5 = 20)
    c1 = feats[0, 1]
    assert c1[0] == 1.0
    assert c1[1] == 0.0
    assert c1[2] == 1.0
    assert c1[4] == 1.0, f"Expected has_arithmetic_error=1, got {c1[4]}"

    # Cand 2: covers only 10, misses 5 -> coverage = 0.5
    c2 = feats[0, 2]
    assert abs(c2[7] - 0.5) < 1e-4, f"Expected coverage=0.5, got {c2[7]}"
    assert c2[8] == 1.0  # missing 1 condition

    # Cand 3: 0 equations, ratio = 1.0 (neutral), coverage = 0.0
    c3 = feats[0, 3]
    assert c3[0] == 0.0
    assert c3[3] == 1.0
    assert c3[4] == 0.0

    print("FEAT-G03 Synthetic Unit Tests PASSED!")


def test_feat_g03_dataset():
    extractor = SymbolicVerificationFeatureExtractor()
    print("Testing FEAT-G03 on validation set (798 samples)...")
    val_bank, val_arrays = load_dataset('validation')

    t0 = time.perf_counter()
    feats = extractor.transform(val_bank, val_arrays)
    elapsed = time.perf_counter() - t0

    assert feats.shape == (798, 4, 14)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Found non-finite values in FEAT-G03!"

    print(f"Validation extraction complete in {elapsed:.3f} s ({(elapsed/798)*1000:.2f} ms/question).")
    print("Feature means across validation candidates:")
    for k in range(14):
        name = extractor.FEATURE_NAMES[k]
        val_mean = feats[:, :, k].mean()
        val_std = feats[:, :, k].std()
        print(f"  {name:28s}: mean = {val_mean:10.4f}, std = {val_std:10.4f}")

    print("FEAT-G03 Dataset Test PASSED!")


if __name__ == "__main__":
    test_feat_g03_unit()
    test_feat_g03_dataset()
