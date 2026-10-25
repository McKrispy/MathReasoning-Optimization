"""Frozen EXP-B17/EXP-B20 inference entrypoint.

Replays the historical three-view soft stack:
1. Arm A embeddings-only attention, seeds 42/43/44, averaged before normalization
2. Arm B embeddings plus 16 structured columns, same seeds
3. Tree R5 on raw 16 structured columns plus 128 SVD features
Then question z-score, historical convex weights, validity/conflict masks and
cluster aggregation at T=0.5.

An explicit checkpoint directory is required. A missing checkpoint raises
instead of falling back to majority voting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence

# Process-start environment is authoritative for OpenMP. These defaults apply
# only when the sealed launcher has not already set them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

DEV_SRC = Path(__file__).resolve().parent
OFFICIAL_CODE = Path(__file__).resolve().parents[2] / "DASE7506-MP2" / "code"

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

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_grad_enabled(False)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class B17InferenceEngine:
    """Encapsulates model weights, SVD projection and config for B17 inference."""

    def __init__(self, checkpoint_dir: Path):
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            self.checkpoint_dir = Path(checkpoint_dir)
            config_path = self.checkpoint_dir / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            self.config = json.loads(config_path.read_text(encoding="utf-8"))

            self.consensus_cols = self.config["features"]["consensus_core"]
            self.symbolic_cols = self.config["features"]["symbolic"]
            self.g01_indices = [
                ConsensusFeatureExtractor.FEATURE_NAMES.index(c) for c in self.consensus_cols
            ]
            self.g03_indices = [
                SymbolicVerificationFeatureExtractor.FEATURE_NAMES.index(c) for c in self.symbolic_cols
            ]

            svd_path = self.checkpoint_dir / "svd_components.npz"
            svd_data = np.load(svd_path)
            self.svd_components = svd_data["components"].astype(np.float32)

            tree_path = self.checkpoint_dir / "tree_r5.pkl"
            with tree_path.open("rb") as handle:
                self.tree_model = pickle.load(handle)

            self.seeds = self.config.get("seeds", [42, 43, 44])
            self.arm_a_models = []
            for seed in self.seeds:
                weights = dict(np.load(self.checkpoint_dir / f"arm_a_seed_{seed}.npz"))
                self.arm_a_models.append(CrossCandidateAttentionNet(weights))

            self.arm_b_models = []
            self.arm_b_means = []
            self.arm_b_stds = []
            for seed in self.seeds:
                weights = dict(np.load(self.checkpoint_dir / f"arm_b_seed_{seed}.npz"))
                self.arm_b_models.append(CrossCandidateAttentionNet(weights))
                self.arm_b_means.append(weights["struct_mean"].astype(np.float32))
                self.arm_b_stds.append(weights["struct_std"].astype(np.float32))

            self.weights = np.array(self.config["weights"], dtype=np.float32)
            self.temperature = float(self.config.get("temperature", 0.5))
            self.mask_penalty = float(self.config.get("mask_penalty", 1e9))

    def predict_bank(self, bank: Sequence[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> List[int]:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            return self._predict_bank(bank, arrays)

    def _predict_bank(self, bank: Sequence[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> List[int]:
        n_questions = len(bank)
        if n_questions == 0:
            return []

        emb = arrays["embeddings"]
        emb_norm = l2_normalize(emb)

        g01_full = ConsensusFeatureExtractor().transform(bank, arrays)
        g03_full = SymbolicVerificationFeatureExtractor().transform(bank, arrays)
        g01 = g01_full[:, :, self.g01_indices].astype(np.float32)
        g03 = g03_full[:, :, self.g03_indices].astype(np.float32)
        struct_16 = np.concatenate([g01, g03], axis=-1)

        flat_emb = emb_norm.reshape(n_questions * 4, 1024)
        proj = (flat_emb @ self.svd_components.T).reshape(n_questions, 4, 64)
        centered = proj - proj.mean(axis=1, keepdims=True)
        svd_128 = np.concatenate([proj, centered], axis=-1).astype(np.float32)

        logits_a_list = [net.forward_numpy(emb_norm) for net in self.arm_a_models]
        view_a = np.mean(logits_a_list, axis=0)

        logits_b_list = []
        for net, mean, std in zip(self.arm_b_models, self.arm_b_means, self.arm_b_stds):
            struct_norm = (struct_16 - mean) / np.maximum(std, 1e-6)
            xb = np.concatenate([emb_norm, struct_norm], axis=-1)
            logits_b_list.append(net.forward_numpy(xb))
        view_b = np.mean(logits_b_list, axis=0)

        x_tree = np.concatenate([struct_16, svd_128], axis=-1)
        flat_tree = x_tree.reshape(n_questions * 4, 144)
        view_tree = np.asarray(
            self.tree_model.decision_function(flat_tree), dtype=np.float32
        ).reshape(n_questions, 4)

        stacked = (
            self.weights[0] * standardize_within_problem(view_a)
            + self.weights[1] * standardize_within_problem(view_b)
            + self.weights[2] * standardize_within_problem(view_tree)
        )
        masked = apply_validity_mask(stacked, bank, penalty=self.mask_penalty)
        masked = apply_conflict_penalty(masked, bank, penalty=self.mask_penalty)
        aggregator = SoftmaxClusterAggregator(temperature=self.temperature)
        chosen = aggregator.aggregate(bank, masked)
        return [int(choice) for choice in chosen]


_CACHED_ENGINE: Optional[B17InferenceEngine] = None
_CACHED_CHECKPOINT: Optional[str] = None


def predict(
    bank: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
    checkpoint: Optional[Path] = None,
) -> List[int]:
    """Score one bank in the order given.

    checkpoint must be the directory that contains the frozen B20 assets.
    """
    global _CACHED_ENGINE, _CACHED_CHECKPOINT

    if common.PROTOCOL != "dase7506-mp2-open-hard-verifier-v5":
        raise RuntimeError(f"Unexpected official protocol: {common.PROTOCOL}")
    if checkpoint is None:
        raise FileNotFoundError("B20 inference requires an explicit checkpoint directory")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    ckpt_str = str(checkpoint)
    if _CACHED_ENGINE is None or _CACHED_CHECKPOINT != ckpt_str:
        _CACHED_ENGINE = B17InferenceEngine(checkpoint)
        _CACHED_CHECKPOINT = ckpt_str
    return _CACHED_ENGINE.predict_bank(bank, arrays)
