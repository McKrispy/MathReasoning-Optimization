"""Direct predict() call for B29 audit comparisons. Not an official runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1]
OFFICIAL = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(SRC))

import common  # type: ignore
import submission  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", default="audit")
    parser.add_argument("--empty", action="store_true")
    args = parser.parse_args()
    if args.empty:
        bank = []
        arrays = {"embeddings": np.zeros((0, 4, 1024), dtype=np.float32)}
        ids = []
    else:
        bank, arrays = common.load_inputs(args.split, args.data_dir)
        ids = [row["id"] for row in bank]
    selected = submission.predict(bank, arrays, args.checkpoint)
    payload = {
        "protocol": common.PROTOCOL,
        "split": "empty" if args.empty else args.split,
        "ids": ids,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
