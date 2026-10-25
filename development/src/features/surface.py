"""FEAT-G02: Surface length, layout, formatting, and wording heuristics extractor.

Extracts length, line formatting, operator density, self-doubt / confidence keywords,
and intra-group comparative rankings to analyze heuristic persuasiveness (supporting RQ2).
"""
import re
from typing import Any, Dict, List
import numpy as np

from .base import BaseFeatureExtractor

# Keywords signaling backtracking, self-doubt, or error correction
DOUBT_PAT = re.compile(
    r'\b(wait|mistake|error|recheck|sorry|alternatively|correcting|wrong|re-evaluat\w*|oops|actually)\b',
    re.IGNORECASE
)

# Keywords signaling definitive conclusions and rhetorical confidence
CONFIDENCE_PAT = re.compile(
    r'\b(clearly|therefore|thus|obviously|definitely|hence|must be|so|in conclusion|proves?|certainly)\b',
    re.IGNORECASE
)

# Step structure and numbered reasoning markers
STEP_PAT = re.compile(
    r'\b(step\s*\d+|first|second|third|fourth|finally|next)\b|^\s*\d+[\.\)]',
    re.IGNORECASE | re.MULTILINE
)

# Arithmetic and relational symbols commonly used in derivations
MATH_SYMBOLS = set("+-*/=%<>^()[]{}")


class SurfaceFeatureExtractor(BaseFeatureExtractor):
    """FEAT-G02: Surface Length, Formatting & Wording Heuristics Extractor.
    
    Features extracted per candidate (16 dimensions):
      [Absolute Surface Features]
       0: char_count (length in characters)
       1: word_count (number of whitespace-separated words)
       2: line_count (number of lines)
       3: newline_ratio (newline characters per total characters)
       4: digit_ratio (digit characters per total characters)
       5: math_symbol_density (math symbols per total characters)
       6: doubt_count (occurrences of doubt/backtracking keywords)
       7: confidence_count (occurrences of confident/conclusive keywords)
       8: step_count (occurrences of step markers / numbered lists)
      
      [Group-Relative Comparative Features across the 4 candidates]
       9: char_count_diff_mean (candidate char count minus group mean)
      10: char_count_rank (rank in group 0..3 by char count)
      11: word_count_diff_mean (candidate word count minus group mean)
      12: word_count_rank (rank in group 0..3 by word count)
      13: doubt_diff_mean (candidate doubt count minus group mean)
      14: confidence_diff_mean (candidate confidence count minus group mean)
      15: math_density_diff_mean (candidate math density minus group mean)
    """

    FEATURE_NAMES = [
        "char_count",
        "word_count",
        "line_count",
        "newline_ratio",
        "digit_ratio",
        "math_symbol_density",
        "doubt_count",
        "confidence_count",
        "step_count",
        "char_count_diff_mean",
        "char_count_rank",
        "word_count_diff_mean",
        "word_count_rank",
        "doubt_diff_mean",
        "confidence_diff_mean",
        "math_density_diff_mean",
    ]

    def __init__(self):
        super().__init__(name="FEAT-G02-surface")

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(bank)
        feats = np.zeros((n, 4, self.feature_dim), dtype=np.float32)

        for i, row in enumerate(bank):
            raw_chars = np.zeros(4, dtype=np.float32)
            raw_words = np.zeros(4, dtype=np.float32)
            raw_lines = np.zeros(4, dtype=np.float32)
            raw_newlines = np.zeros(4, dtype=np.float32)
            raw_digits = np.zeros(4, dtype=np.float32)
            raw_math = np.zeros(4, dtype=np.float32)
            raw_doubt = np.zeros(4, dtype=np.float32)
            raw_conf = np.zeros(4, dtype=np.float32)
            raw_step = np.zeros(4, dtype=np.float32)

            for c in range(4):
                text = row['candidates'][c].get('text', '')
                c_len = len(text)
                denom = max(c_len, 1)

                words = text.split()
                lines = text.splitlines()

                math_count = sum(1 for ch in text if ch in MATH_SYMBOLS)
                digit_count = sum(1 for ch in text if ch.isdigit())
                newline_count = text.count('\n')

                raw_chars[c] = float(c_len)
                raw_words[c] = float(len(words))
                raw_lines[c] = float(len(lines))
                raw_newlines[c] = float(newline_count / denom)
                raw_digits[c] = float(digit_count / denom)
                raw_math[c] = float(math_count / denom)
                raw_doubt[c] = float(len(DOUBT_PAT.findall(text)))
                raw_conf[c] = float(len(CONFIDENCE_PAT.findall(text)))
                raw_step[c] = float(len(STEP_PAT.findall(text)))

            # Absolute features
            feats[i, :, 0] = raw_chars
            feats[i, :, 1] = raw_words
            feats[i, :, 2] = raw_lines
            feats[i, :, 3] = raw_newlines
            feats[i, :, 4] = raw_digits
            feats[i, :, 5] = raw_math
            feats[i, :, 6] = raw_doubt
            feats[i, :, 7] = raw_conf
            feats[i, :, 8] = raw_step

            # Group relative features
            feats[i, :, 9] = raw_chars - raw_chars.mean()
            feats[i, :, 10] = np.argsort(np.argsort(raw_chars)).astype(np.float32)

            feats[i, :, 11] = raw_words - raw_words.mean()
            feats[i, :, 12] = np.argsort(np.argsort(raw_words)).astype(np.float32)

            feats[i, :, 13] = raw_doubt - raw_doubt.mean()
            feats[i, :, 14] = raw_conf - raw_conf.mean()
            feats[i, :, 15] = raw_math - raw_math.mean()

        return feats
