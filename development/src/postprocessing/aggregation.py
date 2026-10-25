"""POST-P02 & POST-P03: Weighted Self-Consistency Cluster Aggregation and Ensemble Gating."""
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore


class WeightedSelfConsistencyAggregator:
    """Aggregates candidate probabilities into final answer clusters:
    
    Score(A_k) = sum_{c in A_k} (p_{i, c}) ^ gamma
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def aggregate(self, bank: List[Dict[str, Any]], probabilities: np.ndarray) -> List[int]:
        """Aggregate probabilities [N, 4] and return optimal candidate index per question."""
        choices: List[int] = []
        n = len(bank)

        for i in range(n):
            row = bank[i]
            probs = probabilities[i]  # [4]
            cluster_scores = defaultdict(float)
            cluster_cands = defaultdict(list)

            for c in range(4):
                ans = common.parse_answer(row['candidates'][c]['text'])
                # If parse fails, treat as unique dummy cluster
                key = ans if ans is not None else f"__unparsed_{c}__"
                score = (float(probs[c]) ** self.gamma) if self.gamma > 0 else 1.0
                cluster_scores[key] += score
                cluster_cands[key].append(c)

            # Highest cluster score, then the more confident member, then the earlier index.
            def cluster_rank(key: str):
                members = cluster_cands[key]
                return (cluster_scores[key], max(float(probs[c]) for c in members), -min(members))

            best_cluster = max(cluster_scores.keys(), key=cluster_rank)
            best_cand = min(cluster_cands[best_cluster], key=lambda c: (-float(probs[c]), c))
            choices.append(int(best_cand))

        return choices


def apply_disagreement_gate(
    choice_m03: np.ndarray,
    choice_m04: np.ndarray,
    confidence_m03: np.ndarray,
    confidence_m04: np.ndarray,
    name: str,
    threshold: float | None = None,
) -> np.ndarray:
    """POST-P03. Agreement keeps the shared index. Disagreement follows the named rule.

    Equal confidence stays with MOD-M03.
    """
    choice_a = np.asarray(choice_m03, dtype=int)
    choice_b = np.asarray(choice_m04, dtype=int)
    if name == "m03_only":
        return choice_a.copy()
    if name == "m04_only":
        return choice_b.copy()
    out = choice_a.copy()
    disagree = choice_a != choice_b
    if name == "higher_confidence":
        take_b = disagree & (np.asarray(confidence_m04) > np.asarray(confidence_m03))
    elif name == "m04_when_m03_below":
        if threshold is None:
            raise ValueError("m04_when_m03_below requires a threshold")
        take_b = disagree & (np.asarray(confidence_m03) < float(threshold))
    else:
        raise ValueError(f"unknown gate {name}")
    out[take_b] = choice_b[take_b]
    return out
