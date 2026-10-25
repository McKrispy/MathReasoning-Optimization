"""Test FEAT-G02: SurfaceFeatureExtractor correctness and dataset profiling."""
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
from features.surface import SurfaceFeatureExtractor


def test_feat_g02_unit():
    extractor = SurfaceFeatureExtractor()
    assert extractor.feature_dim == 16

    mock_bank = [
        {
            "id": "mock_01",
            "question": "What is 2+2?",
            "candidates": [
                {
                    "index": 0,
                    # Short, confident
                    "text": "Clearly, 2 + 2 = 4. Therefore, the answer is 4. #### 4"
                },
                {
                    "index": 1,
                    # Long, multi-step, doubt keyword
                    "text": "Step 1: calculate 2 + 2.\nWait, let me recheck.\n2 + 2 = 4.\nSo it must be 4. #### 4"
                },
                {
                    "index": 2,
                    # Minimal
                    "text": "4 #### 4"
                },
                {
                    "index": 3,
                    # Medium, math symbols
                    "text": "2 + 2 = 5 * 1 - 1 = 4. #### 4"
                },
            ]
        }
    ]
    mock_arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}

    feats = extractor.transform(mock_bank, mock_arrays)
    assert feats.shape == (1, 4, 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()

    # Cand 2 has the shortest length, so its rank in group should be 0.0
    assert feats[0, 2, 10] == 0.0, f"Expected rank 0 for cand 2, got {feats[0, 2, 10]}"
    # Cand 1 has doubt keywords ('wait', 'recheck')
    assert feats[0, 1, 6] >= 2.0, f"Expected doubt count >= 2 for cand 1, got {feats[0, 1, 6]}"
    # Cand 0 has confidence keywords ('clearly', 'therefore')
    assert feats[0, 0, 7] >= 2.0, f"Expected confidence count >= 2 for cand 0, got {feats[0, 0, 7]}"

    print("FEAT-G02 Synthetic Unit Tests PASSED!")


def test_feat_g02_dataset():
    extractor = SurfaceFeatureExtractor()
    print("Testing FEAT-G02 on validation set (798 samples)...")
    val_bank, val_arrays = load_dataset('validation')

    t0 = time.perf_counter()
    feats = extractor.transform(val_bank, val_arrays)
    elapsed = time.perf_counter() - t0

    assert feats.shape == (798, 4, 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Found non-finite values in FEAT-G02!"

    print(f"Validation extraction complete in {elapsed:.3f} s ({(elapsed/798)*1000:.2f} ms/question).")
    print("Feature means across validation candidates:")
    for k in range(16):
        name = extractor.FEATURE_NAMES[k]
        val_mean = feats[:, :, k].mean()
        val_std = feats[:, :, k].std()
        print(f"  {name:25s}: mean = {val_mean:10.4f}, std = {val_std:10.4f}")

    print("FEAT-G02 Dataset Test PASSED!")


if __name__ == "__main__":
    test_feat_g02_unit()
    test_feat_g02_dataset()
