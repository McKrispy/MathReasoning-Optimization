"""Common development utilities for DASE7506-MP2.

Provides dataset loaders, ground-truth label generators,
and Hard subset extractors while respecting official read-only constraints.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure official code directory is in sys.path
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[2] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore # Official loader and parser


def get_data_dir() -> Path:
    """Return the official data directory."""
    return OFFICIAL_CODE_DIR / 'data'


def load_dataset(split: str, data_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Load bank and embeddings array for a given split using official loader."""
    d_dir = data_dir or get_data_dir()
    return common.load_inputs(split, data_dir=d_dir)


def load_gold_references(split: str, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load reference answers for train/validation. Never use test answers for training!"""
    d_dir = data_dir or get_data_dir()
    return common.read_jsonl(d_dir / f'gsm8k_{split}.jsonl')


def build_label_matrix(bank: List[Dict[str, Any]], gold_references: List[Dict[str, Any]]) -> np.ndarray:
    """Construct binary correctness matrix Y of shape [N, 4].
    
    Y[i, c] = 1 if candidate c parses to the correct reference answer, else 0.
    """
    if len(bank) != len(gold_references):
        raise ValueError(f"Bank size ({len(bank)}) and gold size ({len(gold_references)}) mismatch.")
    
    n = len(bank)
    y = np.zeros((n, 4), dtype=np.int32)
    
    for i, (row, gold) in enumerate(zip(bank, gold_references)):
        if row['id'] != gold['id']:
            raise ValueError(f"ID mismatch at index {i}: bank={row['id']} vs gold={gold['id']}")
        gold_ans = common.parse_answer(gold['answer'])
        for c in range(4):
            cand_ans = common.parse_answer(row['candidates'][c]['text'])
            if cand_ans is not None and cand_ans == gold_ans:
                y[i, c] = 1
                
    return y


def get_hard_subset_mask(bank: List[Dict[str, Any]]) -> np.ndarray:
    """Return boolean mask of shape [N] where a question has >= 2 distinct valid parsed answers.
    
    According to audit findings:
    - val_hard: exactly 355 / 798 questions.
    - train_hard: exactly 2114 / 4755 questions.
    """
    n = len(bank)
    mask = np.zeros(n, dtype=bool)
    for i, row in enumerate(bank):
        parsed_answers = set()
        for cand in row['candidates']:
            ans = common.parse_answer(cand['text'])
            if ans is not None:
                parsed_answers.add(ans)
        if len(parsed_answers) >= 2:
            mask[i] = True
    return mask
