"""Unit tests for MOD-M06: DualTowerAttentionModule and DualTowerAttentionNet."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

DEV_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from models.dual_tower import (
    DualTowerAttentionModule,
    DualTowerAttentionNet,
    state_dict_to_numpy,
    uncompressed_weight_bytes,
)
from models.cross_attn_net import loss_components


def test_dual_tower_permutation_equivariance():
    torch.manual_seed(42)
    b, main_dim, aux_dim = 8, 1036, 22
    model = DualTowerAttentionModule(
        main_dim=main_dim,
        aux_dim=aux_dim,
        hidden_dim=128,
        aux_hidden_dim=32,
        num_layers=2,
        num_heads=4,
        input_dropout=0.0,
        attn_dropout=0.0,
    )
    model.eval()

    x_main = torch.randn(b, 4, main_dim)
    x_aux = torch.randn(b, 4, aux_dim)

    scores_orig = model(x_main, x_aux)

    # Test all permutations of 4 candidates
    perm = [2, 0, 3, 1]
    x_main_perm = x_main[:, perm, :]
    x_aux_perm = x_aux[:, perm, :]

    scores_perm = model(x_main_perm, x_aux_perm)
    expected_perm = scores_orig[:, perm]

    max_diff = torch.max(torch.abs(scores_perm - expected_perm)).item()
    assert max_diff < 1e-5, f"Permutation equivariance violated, max diff: {max_diff}"


def test_dual_tower_pytorch_numpy_numerical_equivalence():
    torch.manual_seed(123)
    b, main_dim, aux_dim = 5, 1024, 22
    model = DualTowerAttentionModule(
        main_dim=main_dim,
        aux_dim=aux_dim,
        hidden_dim=128,
        aux_hidden_dim=32,
        num_layers=2,
        num_heads=4,
        input_dropout=0.0,
        attn_dropout=0.0,
    )
    model.eval()

    x_main_t = torch.randn(b, 4, main_dim)
    x_aux_t = torch.randn(b, 4, aux_dim)

    with torch.no_grad():
        scores_torch = model(x_main_t, x_aux_t).numpy()

    weights = state_dict_to_numpy(model)
    net_np = DualTowerAttentionNet(weights)
    scores_np = net_np.forward_numpy(x_main_t.numpy(), x_aux_t.numpy())

    max_diff = np.max(np.abs(scores_torch - scores_np))
    assert max_diff < 1e-5, f"PyTorch vs NumPy forward mismatch: {max_diff}"


def test_dual_tower_numpy_permutation_equivariance():
    torch.manual_seed(456)
    b, main_dim, aux_dim = 6, 1036, 22
    model = DualTowerAttentionModule(main_dim=main_dim, aux_dim=aux_dim)
    weights = state_dict_to_numpy(model)
    net_np = DualTowerAttentionNet(weights)

    rng = np.random.default_rng(42)
    x_m = rng.standard_normal((b, 4, main_dim), dtype=np.float32)
    x_a = rng.standard_normal((b, 4, aux_dim), dtype=np.float32)

    scores_orig = net_np.forward_numpy(x_m, x_a)

    perm = [3, 1, 0, 2]
    scores_perm = net_np.forward_numpy(x_m[:, perm, :], x_a[:, perm, :])
    expected = scores_orig[:, perm]

    max_diff = np.max(np.abs(scores_perm - expected))
    assert max_diff < 1e-5, f"NumPy permutation equivariance violated: {max_diff}"


def test_dual_tower_asset_size():
    model = DualTowerAttentionModule(main_dim=1036, aux_dim=22)
    weights = state_dict_to_numpy(model)
    n_bytes = uncompressed_weight_bytes(weights)
    n_mib = n_bytes / (1024 * 1024)

    # A single model should be ~1.06 MiB, well below 2.0 MiB
    assert n_mib < 2.0
    # 3 seeds combined will be ~3.2 MiB, well below the 16.0 MiB limit
    assert n_mib * 3 < 10.0


def test_dual_tower_loss_computation():
    torch.manual_seed(789)
    b, main_dim, aux_dim = 4, 1036, 22
    model = DualTowerAttentionModule(main_dim=main_dim, aux_dim=aux_dim)
    model.train()

    x_main = torch.randn(b, 4, main_dim)
    x_aux = torch.randn(b, 4, aux_dim)
    labels = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ])

    scores = model(x_main, x_aux)
    listwise, bce = loss_components(scores, labels)
    loss = (listwise + 1.0 * bce).mean()

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss.backward()

    # Check gradients exist
    assert model.proj_main.weight.grad is not None
    assert model.mlp_aux.weight.grad is not None
