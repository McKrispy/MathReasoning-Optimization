"""MOD-M06: Decoupled Dual-Tower Attention Network with In-Layer GroupNorm.

Architectural highlights:
  1. Main Tower (Attention Tower, Semantic Reasoning):
     - Takes candidate embeddings [B, 4, 1024] + optional symbolic reasoning features [B, 4, D_sym].
     - Strictly NO voting consensus features in main tower to avoid gradient starvation.
     - Input dropout (p=0.3) for noise robustness.
     - Projects to hidden dim (128), followed by parameter-free In-Layer Group Normalization:
       delta_{i,c} = LayerNorm(h_{i,c} - (1/4) sum_k h_{i,k})
       Combined: LayerNorm(h_{i,c}) + delta_{i,c}
       Amplifies candidate differential signals without increasing input projection parameters.
     - 2 permutation-equivariant cross-candidate self-attention blocks.
     - Outputs scalar attention score s_{main}.
  2. Auxiliary Tower (Gated Consensus & Verification Tower):
     - Takes FEAT-G01 consensus + FEAT-G03 symbolic + FEAT-G05 fine-grained features [B, 4, D_aux].
     - Lightweight 2-layer MLP with GELU(approximate='tanh') (hidden dim 32).
     - Outputs candidate-level correction bias scalar s_{aux}.
  3. Late Residual Fusion:
     - Final score: s = s_{main} + s_{aux}.
  4. Permutation Equivariant in both PyTorch training and pure NumPy inference.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attn_net import AttentionBlock, loss_components


class DualTowerAttentionModule(nn.Module):
    """PyTorch training module for MOD-M06."""

    def __init__(
        self,
        main_dim: int,
        aux_dim: int,
        hidden_dim: int = 128,
        aux_hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        input_dropout: float = 0.3,
        attn_dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.aux_hidden_dim = aux_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eps = eps

        # --- Main Tower ---
        self.drop_in = nn.Dropout(input_dropout)
        self.proj_main = nn.Linear(main_dim, hidden_dim)
        self.ln_main = nn.LayerNorm(hidden_dim, eps=eps)
        self.ln_diff = nn.LayerNorm(hidden_dim, eps=eps)

        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim, num_heads, attn_dropout, eps) for _ in range(num_layers)]
        )
        self.score_main = nn.Linear(hidden_dim, 1)

        # --- Auxiliary Tower ---
        self.mlp_aux = nn.Linear(aux_dim, aux_hidden_dim)
        self.score_aux = nn.Linear(aux_hidden_dim, 1)

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_main: [B, 4, main_dim] (embeddings + optional G03)
            x_aux:  [B, 4, aux_dim]  (G01 + G03 + G05)
            
        Returns:
            scores: [B, 4] candidate scores
        """
        # 1. Main Tower
        x_m = self.drop_in(x_main)
        h = self.proj_main(x_m)  # [B, 4, hidden_dim]

        # In-Layer Group Instance Normalization (Centering)
        h_mean = h.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        diff = h - h_mean  # [B, 4, hidden_dim]
        h = self.ln_main(h) + self.ln_diff(diff)

        # Self-Attention Blocks
        for block in self.blocks:
            h = block(h)
        s_main = self.score_main(h).squeeze(-1)  # [B, 4]

        # 2. Auxiliary Tower
        h_a = F.gelu(self.mlp_aux(x_aux), approximate="tanh")  # [B, 4, aux_hidden_dim]
        s_aux = self.score_aux(h_a).squeeze(-1)  # [B, 4]

        # 3. Late Residual Fusion
        return s_main + s_aux


def state_dict_to_numpy(module: nn.Module) -> Dict[str, np.ndarray]:
    """Export evaluation weights as float32 numpy arrays."""
    module.eval()
    exported: Dict[str, np.ndarray] = {}
    for key, value in module.state_dict().items():
        exported[key] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    exported["meta_num_heads"] = np.asarray([module.num_heads], dtype=np.int64)
    exported["meta_num_layers"] = np.asarray([module.num_layers], dtype=np.int64)
    exported["meta_eps"] = np.asarray([module.eps], dtype=np.float32)
    exported["meta_main_dim"] = np.asarray([module.main_dim], dtype=np.int64)
    exported["meta_aux_dim"] = np.asarray([module.aux_dim], dtype=np.int64)
    return exported


def uncompressed_weight_bytes(weights: Dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(val).nbytes for val in weights.values()))


def _layer_norm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _multihead_np(x: np.ndarray, wq, bq, wk, bk, wv, bv, wo, bo, num_heads: int) -> np.ndarray:
    batch, length, hidden = x.shape
    head_dim = hidden // num_heads

    def project(weight, bias):
        projected = x @ weight.T + bias
        return projected.reshape(batch, length, num_heads, head_dim).transpose(0, 2, 1, 3)

    query = project(wq, bq)
    key = project(wk, bk)
    value = project(wv, bv)
    logits = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(np.float32(head_dim))
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    attn = exp / exp.sum(axis=-1, keepdims=True)
    context = np.matmul(attn, value).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
    return context @ wo.T + bo


def _gelu_np(x: np.ndarray) -> np.ndarray:
    """Fast, numerically accurate GELU with tanh approximation matching PyTorch."""
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))).astype(np.float32)


class DualTowerAttentionNet:
    """Pure NumPy inference for frozen MOD-M06 weights."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = {k: np.asarray(v) for k, v in weights.items()}
        self.num_heads = int(self.weights["meta_num_heads"][0])
        self.num_layers = int(self.weights["meta_num_layers"][0])
        self.eps = float(self.weights["meta_eps"][0])
        self.main_dim = int(self.weights["meta_main_dim"][0])
        self.aux_dim = int(self.weights["meta_aux_dim"][0])

    def forward_numpy(self, x_main: np.ndarray, x_aux: np.ndarray) -> np.ndarray:
        """Compute candidate scores using pure NumPy matrix operations.
        
        Args:
            x_main: [N, 4, main_dim]
            x_aux:  [N, 4, aux_dim]
            
        Returns:
            scores: [N, 4] float32 array
        """
        if x_main.ndim != 3 or x_main.shape[1] != 4 or x_main.shape[2] != self.main_dim:
            raise ValueError(f"Expected x_main of shape (N, 4, {self.main_dim}), got {x_main.shape}")
        if x_aux.ndim != 3 or x_aux.shape[1] != 4 or x_aux.shape[2] != self.aux_dim:
            raise ValueError(f"Expected x_aux of shape (N, 4, {self.aux_dim}), got {x_aux.shape}")

        w = self.weights

        # 1. Main Tower
        h = x_main.astype(np.float32, copy=False) @ w["proj_main.weight"].T + w["proj_main.bias"]
        h_mean = h.mean(axis=1, keepdims=True)
        diff = h - h_mean
        h = (
            _layer_norm_np(h, w["ln_main.weight"], w["ln_main.bias"], self.eps)
            + _layer_norm_np(diff, w["ln_diff.weight"], w["ln_diff.bias"], self.eps)
        )

        for layer in range(self.num_layers):
            pfx = f"blocks.{layer}."
            mixed = h + _multihead_np(
                h,
                w[pfx + "W_q.weight"], w[pfx + "W_q.bias"],
                w[pfx + "W_k.weight"], w[pfx + "W_k.bias"],
                w[pfx + "W_v.weight"], w[pfx + "W_v.bias"],
                w[pfx + "W_o.weight"], w[pfx + "W_o.bias"],
                self.num_heads,
            )
            h = _layer_norm_np(mixed, w[pfx + "ln.weight"], w[pfx + "ln.bias"], self.eps)

        s_main = (h @ w["score_main.weight"].T + w["score_main.bias"]).squeeze(-1)

        # 2. Auxiliary Tower
        h_a = x_aux.astype(np.float32, copy=False) @ w["mlp_aux.weight"].T + w["mlp_aux.bias"]
        h_a = _gelu_np(h_a)
        s_aux = (h_a @ w["score_aux.weight"].T + w["score_aux.bias"]).squeeze(-1)

        # 3. Late Residual Fusion
        return (s_main + s_aux).astype(np.float32)
