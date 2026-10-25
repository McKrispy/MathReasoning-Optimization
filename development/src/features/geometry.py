"""FEAT-G04: Supplied representations and cross-candidate geometry feature extractor.

Computes semantic centroid alignment, pairwise cosine similarity dispersion,
and intra-cluster vs inter-cluster representational contrast using supplied 1024-d embeddings.
Directly supports comparative ranking in RQ3 and RQ4.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure official code is importable
OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / 'DASE7506-MP2' / 'code'
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.append(str(OFFICIAL_CODE_DIR))

import common  # type: ignore
from .base import BaseFeatureExtractor


class GeometryFeatureExtractor(BaseFeatureExtractor):
    """FEAT-G04: Cross-Candidate 1024-d Geometry Feature Extractor.
    
    Features extracted per candidate (15 dimensions):
      [Centroid & Distance Features]
       0: sim_centroid (cosine similarity to the 4-candidate semantic centroid)
       1: dist_centroid (Euclidean distance to the 4-candidate centroid)
      
      [Pairwise Neighbor Geometry (among the other 3 candidates)]
       2: sim_other_mean (mean cosine similarity to other 3 candidates)
       3: sim_other_max (max cosine similarity to other 3 candidates)
       4: sim_other_min (min cosine similarity to other 3 candidates)
       5: sim_other_std (standard deviation of cosine similarities to other 3 candidates)
      
      [Answer-Aligned Representation Contrast]
       6: has_same_answer_other (1.0 if another candidate shares same parsed answer)
       7: sim_same_answer_mean (mean cosine similarity to same-answer peers, 0.0 if none)
       8: has_diff_answer_other (1.0 if another candidate has a different parsed answer)
       9: sim_diff_answer_mean (mean cosine similarity to different-answer peers, 0.0 if none)
      10: sim_answer_contrast (sim_same_answer_mean minus sim_diff_answer_mean)
      
      [Group-Relative Centroid Rankings]
      11: centroid_sim_diff_mean (sim_centroid minus group mean)
      12: centroid_sim_rank (rank in group 0..3 by centroid similarity; higher is closer to consensus)
      13: dist_centroid_diff_mean (dist_centroid minus group mean)
      14: dist_centroid_rank (rank in group 0..3 by distance to centroid; lower is closer)
    """

    FEATURE_NAMES = [
        "sim_centroid",
        "dist_centroid",
        "sim_other_mean",
        "sim_other_max",
        "sim_other_min",
        "sim_other_std",
        "has_same_answer_other",
        "sim_same_answer_mean",
        "has_diff_answer_other",
        "sim_diff_answer_mean",
        "sim_answer_contrast",
        "centroid_sim_diff_mean",
        "centroid_sim_rank",
        "dist_centroid_diff_mean",
        "dist_centroid_rank",
    ]

    def __init__(self):
        super().__init__(name="FEAT-G04-geometry")

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def transform(self, bank: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> np.ndarray:
        raw_emb = arrays['embeddings']  # [N, 4, 1024]
        n = raw_emb.shape[0]

        # 1. Vectorized L2 normalization
        norms = np.linalg.norm(raw_emb, axis=-1, keepdims=True) + 1e-12
        E = (raw_emb / norms).astype(np.float32)  # [N, 4, 1024]

        # 2. Vectorized Centroid computation
        centroid = E.mean(axis=1, keepdims=True)  # [N, 1, 1024]
        centroid_unit = centroid / (np.linalg.norm(centroid, axis=-1, keepdims=True) + 1e-12)

        # 3. Vectorized Centroid Similarity & Distance
        sim_centroid = np.sum(E * centroid_unit, axis=-1)  # [N, 4]
        dist_centroid = np.linalg.norm(E - centroid, axis=-1)  # [N, 4]

        # 4. Vectorized Pairwise Cosine Similarity Matrix [N, 4, 4]
        S = np.einsum('nij,nkj->nik', E, E)  # S[n, i, k] = cos(E[n, i], E[n, k])

        # Mask off-diagonal (other 3 candidates)
        # S_other shape: [N, 4, 3]
        diag_mask = ~np.eye(4, dtype=bool)
        S_other = S[:, diag_mask].reshape(n, 4, 3)

        sim_other_mean = S_other.mean(axis=-1)
        sim_other_max = S_other.max(axis=-1)
        sim_other_min = S_other.min(axis=-1)
        sim_other_std = S_other.std(axis=-1)

        # 5. Pre-allocate output
        feats = np.zeros((n, 4, self.feature_dim), dtype=np.float32)
        feats[:, :, 0] = sim_centroid
        feats[:, :, 1] = dist_centroid
        feats[:, :, 2] = sim_other_mean
        feats[:, :, 3] = sim_other_max
        feats[:, :, 4] = sim_other_min
        feats[:, :, 5] = sim_other_std

        # 6. Group rankings and diffs
        centroid_diff = sim_centroid - sim_centroid.mean(axis=1, keepdims=True)
        dist_diff = dist_centroid - dist_centroid.mean(axis=1, keepdims=True)

        feats[:, :, 11] = centroid_diff
        feats[:, :, 12] = np.argsort(np.argsort(sim_centroid, axis=1), axis=1).astype(np.float32)
        feats[:, :, 13] = dist_diff
        feats[:, :, 14] = np.argsort(np.argsort(dist_centroid, axis=1), axis=1).astype(np.float32)

        # 7. Answer-aligned representation contrast (combining text parsed answers with embeddings)
        for i in range(n):
            cand_answers = [common.parse_answer(c['text']) for c in bank[i]['candidates']]
            sim_mat = S[i]  # [4, 4]

            for c in range(4):
                ans_c = cand_answers[c]

                same_sims = []
                diff_sims = []

                for k in range(4):
                    if k == c:
                        continue
                    ans_k = cand_answers[k]
                    sim_val = float(sim_mat[c, k])

                    if ans_c is not None and ans_k is not None:
                        if ans_c == ans_k:
                            same_sims.append(sim_val)
                        else:
                            diff_sims.append(sim_val)
                    else:
                        # If either is None, treat as different
                        diff_sims.append(sim_val)

                if same_sims:
                    feats[i, c, 6] = 1.0
                    feats[i, c, 7] = float(np.mean(same_sims))
                else:
                    feats[i, c, 6] = 0.0
                    feats[i, c, 7] = 0.0

                if diff_sims:
                    feats[i, c, 8] = 1.0
                    feats[i, c, 9] = float(np.mean(diff_sims))
                else:
                    feats[i, c, 8] = 0.0
                    feats[i, c, 9] = 0.0

                feats[i, c, 10] = feats[i, c, 7] - feats[i, c, 9]

        return feats
