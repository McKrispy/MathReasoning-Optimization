"""FEAT-G01: Final answer parsing and group consensus feature extractor.

Extracts final numerical answer properties, group agreement/cluster frequency,
majority status, problem controversy, and question condition alignment.
"""
from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set
import numpy as np

# Ensure official code is importable
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from .base import BaseFeatureExtractor


def extract_numbers_from_text(text: str) -> Set[Fraction]:
    """Extract all valid numerical values appearing in question text."""
    matches = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', text)
    numbers = set()
    for m in matches:
        clean = m.replace(',', '')
        try:
            numbers.add(Fraction(clean))
        except (ValueError, ZeroDivisionError):
            pass
    return numbers


class ConsensusFeatureExtractor(BaseFeatureExtractor):
    """FEAT-G01: Question & Final Answer Consensus Feature Extractor.
    
    Features extracted per candidate (11 dimensions):
      0: is_valid (1.0 if answer successfully parsed, else 0.0)
      1: cluster_count (number of candidates in group sharing this exact parsed answer, in 1..4)
      2: cluster_ratio (cluster_count / total valid parsed answers in group)
      3: is_unique_majority (1.0 if answer is the strict unique majority in group)
      4: is_tied_majority (1.0 if answer shares top frequency with another cluster)
      5: num_distinct_answers (number of different valid parsed answers in group)
      6: log_magnitude (log1p(|value|))
      7: is_integer (1.0 if answer is an exact integer)
      8: is_negative (1.0 if value < 0)
      9: is_zero (1.0 if value == 0)
     10: appears_in_question (1.0 if answer value directly matches a number in question text)
    """

    FEATURE_NAMES = [
        "is_valid",
        "cluster_count",
        "cluster_ratio",
        "is_unique_majority",
        "is_tied_majority",
        "num_distinct_answers",
        "log_magnitude",
        "is_integer",
        "is_negative",
        "is_zero",
        "appears_in_question",
    ]

    def __init__(self):
        super().__init__(name="FEAT-G01-consensus")

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(bank)
        feats = np.zeros((n, 4, self.feature_dim), dtype=np.float32)

        for i, row in enumerate(bank):
            cand_answers: List[Optional[str]] = [
                common.parse_answer(c['text']) for c in row['candidates']
            ]
            valid_answers = [a for a in cand_answers if a is not None]
            num_valid = len(valid_answers)

            counts = Counter(valid_answers)
            distinct_answers = len(counts)

            max_count = max(counts.values()) if counts else 0
            majority_answers = {ans for ans, cnt in counts.items() if cnt == max_count and max_count > 0}
            is_unique_maj = (len(majority_answers) == 1 and max_count > 1)
            is_tied_maj = (len(majority_answers) > 1 and max_count > 0)

            # Extract numbers from question
            q_numbers = extract_numbers_from_text(row.get('question', ''))

            for c in range(4):
                ans = cand_answers[c]
                if ans is None:
                    # Non-valid candidate
                    feats[i, c, 5] = float(distinct_answers)
                    continue

                try:
                    frac = Fraction(ans)
                    val = float(frac)
                except (ValueError, ZeroDivisionError):
                    feats[i, c, 5] = float(distinct_answers)
                    continue

                cluster_cnt = counts[ans]
                cluster_rat = cluster_cnt / num_valid if num_valid > 0 else 0.0

                # 0: is_valid
                feats[i, c, 0] = 1.0
                # 1: cluster_count
                feats[i, c, 1] = float(cluster_cnt)
                # 2: cluster_ratio
                feats[i, c, 2] = float(cluster_rat)
                # 3: is_unique_majority
                feats[i, c, 3] = 1.0 if (ans in majority_answers and is_unique_maj) else 0.0
                # 4: is_tied_majority
                feats[i, c, 4] = 1.0 if (ans in majority_answers and is_tied_maj) else 0.0
                # 5: num_distinct_answers
                feats[i, c, 5] = float(distinct_answers)
                # 6: log_magnitude
                # Safe magnitude clamping to prevent overflow
                abs_val = min(abs(val), 1e12)
                feats[i, c, 6] = float(math.log1p(abs_val))
                # 7: is_integer
                feats[i, c, 7] = 1.0 if frac.denominator == 1 else 0.0
                # 8: is_negative
                feats[i, c, 8] = 1.0 if frac < 0 else 0.0
                # 9: is_zero
                feats[i, c, 9] = 1.0 if frac == 0 else 0.0
                # 10: appears_in_question
                feats[i, c, 10] = 1.0 if frac in q_numbers else 0.0

        return feats
