"""Training pipeline entrypoint for DASE7506-MP2.

Handles feature extraction, model fitting, and checkpoint export.
"""
import argparse
from pathlib import Path
import sys

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from common_utils import load_dataset, load_gold_references, build_label_matrix, get_hard_subset_mask


def main():
    parser = argparse.ArgumentParser(description="Train mathematical reasoning reranker.")
    parser.add_argument("--model", choices=["baseline", "pointwise", "cross_attn", "gbdt"], default="cross_attn")
    parser.add_argument("--output-bundle", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "model_bundle")
    args = parser.parse_args()

    print(f"Loading training data...")
    bank, arrays = load_dataset("train")
    gold = load_gold_references("train")
    y = build_label_matrix(bank, gold)
    hard_mask = get_hard_subset_mask(bank)

    print(f"Loaded {len(bank)} train samples. Hard subset has {hard_mask.sum()} samples.")
    print(f"Model {args.model} training pipeline initialized.")


if __name__ == "__main__":
    main()
