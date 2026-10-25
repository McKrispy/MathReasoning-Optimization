"""INFRA-02 Test: Verify ground-truth label generation and Hard subset counts."""
from pathlib import Path
import sys
import numpy as np

DEV_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = DEV_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OFFICIAL_CODE_DIR = DEV_ROOT.parent / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from common_utils import load_dataset, load_gold_references, build_label_matrix, get_hard_subset_mask


def test_infra_labels_and_hard_subsets():
    print("Testing INFRA-02 on validation split...")
    val_bank, val_arrays = load_dataset('validation')
    val_gold = load_gold_references('validation')
    val_y = build_label_matrix(val_bank, val_gold)
    val_hard = get_hard_subset_mask(val_bank)

    assert len(val_bank) == 798, f"Expected 798 val items, got {len(val_bank)}"
    assert val_y.shape == (798, 4), f"Wrong val label shape: {val_y.shape}"
    assert val_hard.sum() == 355, f"Expected 355 val_hard items, got {val_hard.sum()}"
    # Verify every group has at least one correct answer
    assert (val_y.sum(axis=1) >= 1).all(), "Found validation sample with 0 correct answers!"

    print(f"Validation verified: N={len(val_bank)}, val_hard={val_hard.sum()} (44.5%), all have >=1 correct candidate.")

    print("Testing INFRA-02 on train split...")
    train_bank, train_arrays = load_dataset('train')
    train_gold = load_gold_references('train')
    train_y = build_label_matrix(train_bank, train_gold)
    train_hard = get_hard_subset_mask(train_bank)

    assert len(train_bank) == 4755, f"Expected 4755 train items, got {len(train_bank)}"
    assert train_y.shape == (4755, 4), f"Wrong train label shape: {train_y.shape}"
    assert train_hard.sum() == 2114, f"Expected 2114 train_hard items, got {train_hard.sum()}"
    assert (train_y.sum(axis=1) >= 1).all(), "Found train sample with 0 correct answers!"

    print(f"Train verified: N={len(train_bank)}, train_hard={train_hard.sum()} (44.5%), all have >=1 correct candidate.")
    print("INFRA-02 VERIFICATION SUCCESSFUL!")


if __name__ == '__main__':
    test_infra_labels_and_hard_subsets()
