"""EXP-B20 candidate inference entrypoint.

Exposes predict(bank, arrays, checkpoint) conforming to the official specification.
Replays the EXP-B17 3-View Soft Stacking system:
1. View 1: Arm A (embeddings-only permutation-equivariant cross-candidate attention, 3-seed average)
2. View 2: Arm B (embeddings + 16 structured reasoning features, 3-seed average)
3. View 3: Tree R5 (HistGradientBoostingClassifier on 16 structured + 128 SVD features)
Followed by:
- Within-problem logit standardization
- Convex combination soft stacking with weights [0.6520, 0.2774, 0.0706]
- POST-P05 hard validity masking & parser conflict penalty
- POST-P06 within-problem Softmax cluster aggregation (T=0.5)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Resolve dev/src directory so module can locate project dependencies
DEV_SRC = Path(__file__).resolve().parents[2] / "src"
OFFICIAL_CODE = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"

if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))
if str(OFFICIAL_CODE) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE))

import common  # type: ignore
from features.consensus import ConsensusFeatureExtractor
from features.symbolic import SymbolicVerificationFeatureExtractor
from models.cross_attn_net import CrossCandidateAttentionNet
from postprocessing.masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
)
from postprocessing.soft_stacking import standardize_within_problem


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class B17InferenceEngine:
    """Encapsulates model weights, SVD projection and config for B17 inference."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        # Column indices
        self.consensus_cols = self.config["features"]["consensus_core"]
        self.symbolic_cols = self.config["features"]["symbolic"]
        self.g01_indices = [
            ConsensusFeatureExtractor.FEATURE_NAMES.index(c) for c in self.consensus_cols
        ]
        self.g03_indices = [
            SymbolicVerificationFeatureExtractor.FEATURE_NAMES.index(c) for c in self.symbolic_cols
        ]

        # SVD projection matrix (64, 1024)
        svd_path = self.checkpoint_dir / "svd_components.npz"
        svd_data = np.load(svd_path)
        self.svd_components = svd_data["components"].astype(np.float32)

        # Tree model
        tree_path = self.checkpoint_dir / "tree_r5.pkl"
        with tree_path.open("rb") as handle:
            self.tree_model = pickle.load(handle)

        # Cross-candidate attention models
        self.seeds = self.config.get("seeds", [42, 43, 44])
        self.arm_a_models = []
        for s in self.seeds:
            wa = dict(np.load(self.checkpoint_dir / f"arm_a_seed_{s}.npz"))
            self.arm_a_models.append(CrossCandidateAttentionNet(wa))

        self.arm_b_models = []
        self.arm_b_means = []
        self.arm_b_stds = []
        for s in self.seeds:
            wb = dict(np.load(self.checkpoint_dir / f"arm_b_seed_{s}.npz"))
            self.arm_b_models.append(CrossCandidateAttentionNet(wb))
            self.arm_b_means.append(wb["struct_mean"].astype(np.float32))
            self.arm_b_stds.append(wb["struct_std"].astype(np.float32))

        # Stacking weights & post-processing settings
        self.weights = np.array(self.config["weights"], dtype=np.float32)
        self.temperature = float(self.config.get("temperature", 0.5))
        self.mask_penalty = float(self.config.get("mask_penalty", 1e9))

    def predict_bank(self, bank: Sequence[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> List[int]:
        n_questions = len(bank)
        if n_questions == 0:
            return []

        # 1. Normalized candidate embeddings [N, 4, 1024]
        emb = arrays["embeddings"]
        emb_norm = l2_normalize(emb)

        # 2. Extract structured reasoning features (16 cols)
        g01_full = ConsensusFeatureExtractor().transform(bank, arrays)
        g03_full = SymbolicVerificationFeatureExtractor().transform(bank, arrays)
        g01 = g01_full[:, :, self.g01_indices].astype(np.float32)
        g03 = g03_full[:, :, self.g03_indices].astype(np.float32)
        struct_16 = np.concatenate([g01, g03], axis=-1)  # [N, 4, 16]

        # 3. SVD features (64 proj + 64 centered = 128 cols)
        flat_emb = emb_norm.reshape(n_questions * 4, 1024)
        proj = (flat_emb @ self.svd_components.T).reshape(n_questions, 4, 64)
        centered = proj - proj.mean(axis=1, keepdims=True)
        svd_128 = np.concatenate([proj, centered], axis=-1).astype(np.float32)

        # 4. View 1: Arm A (pure geometric semantic attention, 3-seed average)
        logits_a_list = [net.forward_numpy(emb_norm) for net in self.arm_a_models]
        v1 = np.mean(logits_a_list, axis=0)  # [N, 4]

        # 5. View 2: Arm B (symbolic reasoning + consensus attention, 3-seed average)
        logits_b_list = []
        for net, m, s in zip(self.arm_b_models, self.arm_b_means, self.arm_b_stds):
            struct_norm = (struct_16 - m) / np.maximum(s, 1e-6)
            xb = np.concatenate([emb_norm, struct_norm], axis=-1)
            logits_b_list.append(net.forward_numpy(xb))
        v2 = np.mean(logits_b_list, axis=0)  # [N, 4]

        # 6. View 3: Tree R5 (HistGradientBoosting on raw struct_16 + svd_128)
        x_tree = np.concatenate([struct_16, svd_128], axis=-1)  # [N, 4, 144]
        flat_tree = x_tree.reshape(n_questions * 4, 144)
        v4 = np.asarray(self.tree_model.decision_function(flat_tree), dtype=np.float32).reshape(n_questions, 4)

        # 7. Within-problem Standardization & Convex Soft Stacking
        s1 = standardize_within_problem(v1)
        s2 = standardize_within_problem(v2)
        s4 = standardize_within_problem(v4)
        stacked = self.weights[0] * s1 + self.weights[1] * s2 + self.weights[2] * s4

        # 8. POST-P05: Hard validity mask and heading conflict penalty
        z_masked = apply_validity_mask(stacked, bank, penalty=self.mask_penalty)
        z_masked = apply_conflict_penalty(z_masked, bank, penalty=self.mask_penalty)

        # 9. POST-P06: Within-problem Softmax cluster aggregation
        aggregator = SoftmaxClusterAggregator(temperature=self.temperature)
        chosen = aggregator.aggregate(bank, z_masked)
        return [int(c) for c in chosen]


_CACHED_ENGINE: Optional[B17InferenceEngine] = None
_CACHED_CHECKPOINT: Optional[str] = None


def predict(
    bank: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
    checkpoint: Optional[Path] = None,
) -> List[int]:
    """Official prediction interface.
    
    bank: question dictionaries with 'id', 'question', 'candidates'
    arrays['embeddings']: float32 array of shape [N, 4, 1024]
    checkpoint: Path to directory containing assets, or None for default
    """
    global _CACHED_ENGINE, _CACHED_CHECKPOINT

    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / "model_assets"
    else:
        checkpoint = Path(checkpoint).resolve()

    ckpt_str = str(checkpoint)
    if _CACHED_ENGINE is None or _CACHED_CHECKPOINT != ckpt_str:
        _CACHED_ENGINE = B17InferenceEngine(checkpoint)
        _CACHED_CHECKPOINT = ckpt_str

    return _CACHED_ENGINE.predict_bank(bank, arrays)
