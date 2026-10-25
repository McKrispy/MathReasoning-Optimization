"""Offline evaluation and ablation analysis entrypoint.

Evaluates predictions on val_all, val_hard, and computes detailed error breakdowns.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure src and official code are in path
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

import common  # type: ignore
from common_utils import load_dataset, load_gold_references, build_label_matrix, get_hard_subset_mask
from submission import predict


def evaluate_split(split: str = "validation", checkpoint: Path = None):
    bank, arrays = load_dataset(split)
    gold = load_gold_references(split)
    y = build_label_matrix(bank, gold)
    hard_mask = get_hard_subset_mask(bank)

    choices = predict(bank, arrays, checkpoint)
    
    # Compute accuracy
    correct = [y[i, choices[i]] == 1 for i in range(len(bank))]
    acc_all = 100.0 * np.mean(correct)
    acc_hard = 100.0 * np.mean([correct[i] for i in range(len(bank)) if hard_mask[i]])

    print(f"=== Evaluation on {split} ===")
    print(f"All samples (n={len(bank)}): {acc_all:.2f}%")
    print(f"Hard subset (n={hard_mask.sum()}): {acc_hard:.2f}%")
    return {"acc_all": acc_all, "acc_hard": acc_hard}


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction method.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    evaluate_split(args.split, args.checkpoint)


if __name__ == "__main__":
    main()
