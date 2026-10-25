"""Test FEAT-G04: GeometryFeatureExtractor correctness and dataset profiling."""
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
from features.geometry import GeometryFeatureExtractor


def test_feat_g04_unit():
    extractor = GeometryFeatureExtractor()
    assert extractor.feature_dim == 15

    # Case 1: 4 orthogonal basis vectors
    mock_bank = [
        {
            "id": "mock_01",
            "question": "Dummy question",
            "candidates": [
                {"index": 0, "text": "Answer 10 #### 10"},
                {"index": 1, "text": "Answer 10 #### 10"}, # same answer as 0
                {"index": 2, "text": "Answer 20 #### 20"}, # different answer
                {"index": 3, "text": "Answer 30 #### 30"}, # different answer
            ]
        }
    ]
    # Orthogonal embeddings in 4 dimensions (padded to 1024)
    emb = np.zeros((1, 4, 1024), dtype=np.float32)
    for c in range(4):
        emb[0, c, c] = 1.0

    mock_arrays = {"embeddings": emb}
    feats = extractor.transform(mock_bank, mock_arrays)

    assert feats.shape == (1, 4, 15)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()

    # Orthogonal vectors:
    # Centroid has 0.25 in first 4 dims. Norm is 0.5. Unit centroid has 0.5 in first 4 dims.
    # Dot product of e_c with unit centroid is 0.5
    for c in range(4):
        assert abs(feats[0, c, 0] - 0.5) < 1e-4, f"Expected sim_centroid=0.5, got {feats[0, c, 0]}"
        # Pairwise cosine similarity to other candidates should be 0.0
        assert abs(feats[0, c, 2] - 0.0) < 1e-4, f"Expected sim_other_mean=0.0, got {feats[0, c, 2]}"

    # Cand 0 has cand 1 with same answer -> has_same_answer_other = 1.0
    assert feats[0, 0, 6] == 1.0
    # Cand 2 has no other candidate with same answer -> has_same_answer_other = 0.0
    assert feats[0, 2, 6] == 0.0

    print("FEAT-G04 Synthetic Unit Tests PASSED!")


def test_feat_g04_dataset():
    extractor = GeometryFeatureExtractor()
    print("Testing FEAT-G04 on validation set (798 samples)...")
    val_bank, val_arrays = load_dataset('validation')

    t0 = time.perf_counter()
    feats = extractor.transform(val_bank, val_arrays)
    elapsed = time.perf_counter() - t0

    assert feats.shape == (798, 4, 15)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "Found non-finite values in FEAT-G04!"

    print(f"Validation extraction complete in {elapsed:.3f} s ({(elapsed/798)*1000:.2f} ms/question).")
    print("Feature means across validation candidates:")
    for k in range(15):
        name = extractor.FEATURE_NAMES[k]
        val_mean = feats[:, :, k].mean()
        val_std = feats[:, :, k].std()
        print(f"  {name:26s}: mean = {val_mean:10.4f}, std = {val_std:10.4f}")

    print("FEAT-G04 Dataset Test PASSED!")


if __name__ == "__main__":
    test_feat_g04_unit()
    test_feat_g04_dataset()
