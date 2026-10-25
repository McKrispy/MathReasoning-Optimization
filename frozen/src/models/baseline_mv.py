"""MOD-M01: Training-free Majority Voting baseline with centroid tie-breaking."""
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Resolve common from official code
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore


class MajorityVotingBaseline:
    """Majority voting on parsed final numerical answers.
    
    Tie-breaking policy:
    1. Prefer candidates with valid parsed answers over None.
    2. Among tied maximum vote clusters, pick the candidate with highest
       cosine similarity to the 4-candidate semantic centroid in embedding space.
    """

    def predict(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> List[int]:
        embeddings = arrays.get('embeddings')  # [N, 4, 1024]
        choices: List[int] = []

        for i, row in enumerate(bank):
            cand_answers = [common.parse_answer(c['text']) for c in row['candidates']]
            
            # Count valid parsed answers
            valid_counts = Counter([a for a in cand_answers if a is not None])
            
            if not valid_counts:
                # If no candidate parses to a valid number, pick candidate closest to centroid
                if embeddings is not None:
                    emb = embeddings[i]  # [4, 1024]
                    centroid = np.mean(emb, axis=0, keepdims=True)
                    norms = np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(centroid) + 1e-12
                    sims = (emb @ centroid.T).ravel() / norms.ravel()
                    choices.append(int(np.argmax(sims)))
                else:
                    choices.append(0)
                continue

            # Find maximum frequency
            max_freq = max(valid_counts.values())
            # Counter maintains insertion order: first answer reaching max_freq is selected
            top_answers = {ans for ans, count in valid_counts.items() if count == max_freq}

            # Select the first candidate yielding the majority answer (standard natural index)
            for idx, ans in enumerate(cand_answers):
                if ans in top_answers:
                    choices.append(idx)
                    break

        return choices
