"""FEAT-G06: character 3/4-gram hashed TF-IDF.

The transform is an ordinary text statistic. It is not a process reward, a
likelihood ratio, or a reference-model score. Rules below are fixed; they are
not edited from validation errors.

Normalization, in order:
1. Lowercase.
2. Collapse every whitespace run to one space and strip the ends.
3. Replace a numeric span with the single character ``0``. A span is one or
   more digits with optional internal comma or dot groups, such as ``60``,
   ``2.50`` and ``1,000``. Operators and other punctuation stay, so ``1/4``
   becomes ``0/0``.

Synthetic results of that rule:
- ``The TOTAL is 60 km/h.`` -> ``the total is 0 km/h.``
- ``Wait  2.50\\nmore`` -> ``wait 0 more``
- ``1,000 apples and 1/4.`` -> ``0 apples and 0/0.``

Character 3-grams and 4-grams are hashed with unsigned CRC32 modulo 4096.
``tf = 1 + ln(count)`` for a nonzero bucket, ``idf = ln((1+n_fit)/(1+df)) + 1``,
then each candidate row is L2-normalized. ``n_fit`` and ``df`` count candidates.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import sparse

DIMENSIONS = 4096
NGRAMS = (3, 4)
NUMBER_PLACEHOLDER = "0"
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_candidate_text(text: str) -> str:
    collapsed = _WHITESPACE.sub(" ", text.lower()).strip()
    return _NUMBER.sub(NUMBER_PLACEHOLDER, collapsed)


def hash_bucket(ngram: str, dimensions: int = DIMENSIONS) -> int:
    digest = zlib.crc32(ngram.encode("utf-8")) & 0xFFFFFFFF
    return int(digest % dimensions)


def hashed_counts(text: str, ngrams: Sequence[int] = NGRAMS, dimensions: int = DIMENSIONS) -> Dict[int, int]:
    normalized = normalize_candidate_text(text)
    counts: Dict[int, int] = {}
    for width in ngrams:
        if len(normalized) < width:
            continue
        for start in range(len(normalized) - width + 1):
            bucket = hash_bucket(normalized[start:start + width], dimensions)
            counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def idf_from_counts(counts: Sequence[Mapping[int, int]], dimensions: int = DIMENSIONS) -> np.ndarray:
    """Fit IDF on these candidates only. df is the number of candidates with a nonzero bucket."""
    document_frequency = np.zeros(dimensions, dtype=np.float64)
    for row in counts:
        for bucket in row:
            document_frequency[bucket] += 1.0
    n_fit = float(len(counts))
    return np.log((1.0 + n_fit) / (1.0 + document_frequency)) + 1.0


def tfidf_matrix(counts: Sequence[Mapping[int, int]], idf: np.ndarray) -> sparse.csr_matrix:
    """Sparse L2-normalized TF-IDF. Empty candidates stay zero rows."""
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    idf_values = np.asarray(idf, dtype=np.float64)
    for row in counts:
        if not row:
            indptr.append(len(indices))
            continue
        buckets = list(row.keys())
        values = [(1.0 + math.log(row[bucket])) * float(idf_values[bucket]) for bucket in buckets]
        norm = math.sqrt(sum(value * value for value in values))
        if norm > 0.0:
            values = [value / norm for value in values]
        order = sorted(range(len(buckets)), key=buckets.__getitem__)
        for position in order:
            indices.append(int(buckets[position]))
            data.append(float(values[position]))
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(len(counts), int(idf_values.shape[0])),
    )


def linear_scores(matrix: sparse.csr_matrix, coefficients: np.ndarray, intercept: float) -> np.ndarray:
    """NumPy/SciPy sparse dot used at inference. No training framework required."""
    scores = matrix @ np.asarray(coefficients, dtype=np.float64).reshape(-1)
    return np.asarray(scores, dtype=np.float64).reshape(-1) + float(intercept)


class CharHashTfidf:
    """Fold-local vectorizer. Fit on one candidate list, then transform another."""

    def __init__(self, dimensions: int = DIMENSIONS, ngrams: Sequence[int] = NGRAMS):
        self.dimensions = int(dimensions)
        self.ngrams = tuple(int(width) for width in ngrams)
        self.idf_: np.ndarray | None = None
        self.n_fit_: int = 0

    def fit_counts(self, counts: Sequence[Mapping[int, int]]) -> "CharHashTfidf":
        self.n_fit_ = int(len(counts))
        self.idf_ = idf_from_counts(counts, self.dimensions).astype(np.float64)
        return self

    def transform_counts(self, counts: Sequence[Mapping[int, int]]) -> sparse.csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("CharHashTfidf.fit_counts must run before transform_counts")
        return tfidf_matrix(counts, self.idf_)


def candidate_texts(bank: Sequence[Mapping]) -> List[str]:
    """Candidate strings in bank order. Each row's candidates must already be indexes 0, 1, 2, 3."""
    texts: List[str] = []
    for row in bank:
        candidates = row["candidates"]
        if [int(candidate["index"]) for candidate in candidates] != [0, 1, 2, 3]:
            raise RuntimeError("candidate list is not in index order 0..3")
        texts.extend(str(candidate["text"]) for candidate in candidates)
    return texts


def count_candidates(texts: Iterable[str]) -> List[Dict[int, int]]:
    return [hashed_counts(text) for text in texts]
