"""MOD-M03: Permutation-equivariant cross-candidate attention.

Training uses PyTorch. Frozen inference is a pure NumPy forward of the exported
float32 weights, with no positional encoding so a permutation of the four
candidates permutes the scores the same way.
"""
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """One post-norm self-attention block. No positional encoding."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, eps: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        q = self.W_q(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, hidden)
        out = self.drop(self.W_o(ctx))
        return self.ln(x + out)


class CrossCandidateAttentionModule(nn.Module):
    """[B, 4, D] -> [B, 4] scores. Equivariant to permutations of the 4 candidates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eps = eps
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim, num_heads, dropout, eps) for _ in range(num_layers)]
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return self.score(h).squeeze(-1)


def multi_positive_loss(scores: torch.Tensor, labels: torch.Tensor, lam: float) -> torch.Tensor:
    """Listwise multi-positive log-likelihood plus summed BCE.

    L = -log(sum_{c in C+} exp(s_c) / sum_k exp(s_k)) + λ * sum_k BCE(σ(s_k), y_k)
    Every row must contain at least one positive label.
    """
    if scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError(f"scores/labels shape mismatch: {tuple(scores.shape)} vs {tuple(labels.shape)}")
    listwise, bce = loss_components(scores, labels)
    return (listwise + lam * bce).mean()


def loss_components(scores: torch.Tensor, labels: torch.Tensor):
    """Per-row listwise term and per-row summed BCE, before the batch mean."""
    positive = labels >= 0.5
    if not bool(positive.any(dim=-1).all()):
        raise ValueError("multi-positive listwise loss requires at least one positive label per row")
    log_denom = torch.logsumexp(scores, dim=-1)
    masked = scores.masked_fill(~positive, float("-inf"))
    log_numer = torch.logsumexp(masked, dim=-1)
    listwise = -(log_numer - log_denom)
    bce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none").sum(dim=-1)
    return listwise, bce


def state_dict_to_numpy(module: nn.Module) -> Dict[str, np.ndarray]:
    """Export eval weights as float32 arrays. Dropout buffers are not stored."""
    module.eval()
    exported: Dict[str, np.ndarray] = {}
    for key, value in module.state_dict().items():
        exported[key] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    exported["meta_num_heads"] = np.asarray([module.num_heads], dtype=np.int64)
    exported["meta_num_layers"] = np.asarray([module.num_layers], dtype=np.int64)
    exported["meta_eps"] = np.asarray([module.eps], dtype=np.float32)
    return exported


def uncompressed_weight_bytes(weights: Dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(value).nbytes for value in weights.values()))


class CrossCandidateAttentionNet:
    """NumPy inference for a frozen MOD-M03 checkpoint."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.weights = {key: np.asarray(value) for key, value in weights.items()}
        self.num_heads = int(self.weights["meta_num_heads"][0])
        self.num_layers = int(self.weights["meta_num_layers"][0])
        self.eps = float(self.weights["meta_eps"][0])
        hidden = int(self.weights["proj.weight"].shape[0])
        if hidden % self.num_heads != 0:
            raise ValueError("exported hidden dim is not divisible by num_heads")

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """Score candidates.

        Args:
            x: float array [N, 4, D]. D must match the exported projection.
        Returns:
            scores: float32 array [N, 4].
        """
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError(f"expected [N, 4, D], got {tuple(x.shape)}")
        expected = int(self.weights["proj.weight"].shape[1])
        if x.shape[-1] != expected:
            raise ValueError(f"input dim {x.shape[-1]} does not match checkpoint dim {expected}")
        h = x.astype(np.float32, copy=False) @ self.weights["proj.weight"].T + self.weights["proj.bias"]
        for layer in range(self.num_layers):
            h = self._block(h, f"blocks.{layer}.")
        scores = h @ self.weights["score.weight"].T + self.weights["score.bias"]
        return scores.squeeze(-1).astype(np.float32)

    def _block(self, x: np.ndarray, prefix: str) -> np.ndarray:
        weights = self.weights
        mixed = x + _multihead(
            x,
            weights[prefix + "W_q.weight"], weights[prefix + "W_q.bias"],
            weights[prefix + "W_k.weight"], weights[prefix + "W_k.bias"],
            weights[prefix + "W_v.weight"], weights[prefix + "W_v.bias"],
            weights[prefix + "W_o.weight"], weights[prefix + "W_o.bias"],
            self.num_heads,
        )
        return _layer_norm(mixed, weights[prefix + "ln.weight"], weights[prefix + "ln.bias"], self.eps)


def _multihead(x, wq, bq, wk, bk, wv, bv, wo, bo, num_heads: int) -> np.ndarray:
    batch, length, hidden = x.shape
    head_dim = hidden // num_heads

    def project(weight, bias):
        projected = x @ weight.T + bias
        return projected.reshape(batch, length, num_heads, head_dim).transpose(0, 2, 1, 3)

    query = project(wq, bq)
    key = project(wk, bk)
    value = project(wv, bv)
    logits = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(np.float32(head_dim))
    attn = _softmax(logits)
    context = np.matmul(attn, value).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
    return context @ wo.T + bo


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta
