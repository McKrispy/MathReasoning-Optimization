"""MOD-M05: answer-aware cross-candidate attention.

MOD-M03 plus one learnable scalar per layer and head added to the attention
logits of candidate pairs that share a valid parsed answer:

    logit_{jk} += beta_{l,h} * S_{jk},   S_{jk} = 1[j != k and a_j == a_k valid]

beta starts at 0, so training starts from MOD-M03. S is permuted with the
candidates, so the model stays permutation-equivariant.
"""
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from models.cross_attn_net import CrossCandidateAttentionNet, _layer_norm, _softmax

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[3] / "DASE7506-MP2" / "code"
if str(OFFICIAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_DIR))

import common  # type: ignore  # noqa: E402


def same_answer_matrix(bank: List[Dict[str, Any]]) -> np.ndarray:
    """[N, 4, 4] float32; 1 where two different candidates parse to the same valid answer."""
    out = np.zeros((len(bank), 4, 4), dtype=np.float32)
    for i, row in enumerate(bank):
        answers = [common.parse_answer(c["text"]) for c in row["candidates"]]
        for j in range(4):
            for k in range(4):
                if j != k and answers[j] is not None and answers[j] == answers[k]:
                    out[i, j, k] = 1.0
    return out


class AnswerAwareBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.zeros(num_heads))
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        q = self.W_q(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        logits = logits + self.beta.view(1, -1, 1, 1) * same.unsqueeze(1)
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.ln(x + self.drop(self.W_o(ctx)))


class AnswerAwareAttentionModule(nn.Module):
    """([B, 4, D], [B, 4, 4]) -> [B, 4] scores."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eps = eps
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([AnswerAwareBlock(hidden_dim, num_heads, dropout, eps) for _ in range(num_layers)])
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, same: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h, same)
        return self.score(h).squeeze(-1)


class AnswerAwareAttentionNet(CrossCandidateAttentionNet):
    """NumPy inference for a frozen MOD-M05 checkpoint."""

    def forward_numpy(self, x: np.ndarray, same: np.ndarray) -> np.ndarray:
        if same.shape != (x.shape[0], 4, 4):
            raise ValueError(f"same-answer matrix shape {same.shape} does not match {x.shape}")
        self._same = same.astype(np.float32, copy=False)
        return super().forward_numpy(x)

    def _block(self, x: np.ndarray, prefix: str) -> np.ndarray:
        w = self.weights
        batch, length, hidden = x.shape
        heads = self.num_heads
        head_dim = hidden // heads

        def project(name):
            out = x @ w[prefix + name + ".weight"].T + w[prefix + name + ".bias"]
            return out.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = project("W_q"), project("W_k"), project("W_v")
        logits = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(np.float32(head_dim))
        logits = logits + w[prefix + "beta"].reshape(1, -1, 1, 1) * self._same[:, None, :, :]
        ctx = np.matmul(_softmax(logits), v).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        mixed = x + ctx @ w[prefix + "W_o.weight"].T + w[prefix + "W_o.bias"]
        return _layer_norm(mixed, w[prefix + "ln.weight"], w[prefix + "ln.bias"], self.eps)
