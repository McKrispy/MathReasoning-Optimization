"""Fast checks for MOD-M03. The full EXP-B07 fit is the experiment script, not this test."""
from pathlib import Path
import sys
import numpy as np
import torch

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from models.cross_attn_net import (
    CrossCandidateAttentionModule,
    CrossCandidateAttentionNet,
    multi_positive_loss,
    state_dict_to_numpy,
    uncompressed_weight_bytes,
)


def _toy_module(seed: int = 0) -> CrossCandidateAttentionModule:
    torch.manual_seed(seed)
    return CrossCandidateAttentionModule(
        input_dim=16, hidden_dim=32, num_layers=2, num_heads=4, dropout=0.1, eps=1e-5,
    )


def test_numpy_matches_torch_eval():
    module = _toy_module()
    module.eval()
    weights = state_dict_to_numpy(module)
    features = np.random.default_rng(1).normal(size=(5, 4, 16)).astype(np.float32)
    with torch.no_grad():
        torch_scores = module(torch.from_numpy(features)).numpy()
    numpy_scores = CrossCandidateAttentionNet(weights).forward_numpy(features)
    gap = float(np.max(np.abs(torch_scores - numpy_scores)))
    assert gap < 1e-5, gap


def test_permutation_equivariance():
    module = _toy_module(2)
    module.eval()
    weights = state_dict_to_numpy(module)
    net = CrossCandidateAttentionNet(weights)
    features = np.random.default_rng(3).normal(size=(6, 4, 16)).astype(np.float32)
    order = np.array([2, 0, 3, 1])
    original = net.forward_numpy(features)
    permuted = net.forward_numpy(features[:, order, :])
    gap = float(np.max(np.abs(original[:, order] - permuted)))
    assert gap < 1e-5, gap


def test_listwise_loss_matches_hand_calculation():
    scores = torch.zeros(1, 4)
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss = multi_positive_loss(scores, labels, lam=1.0)
    expected = -np.log(0.25) + 4.0 * np.log(2.0)
    assert abs(float(loss) - expected) < 1e-5


def test_exported_full_width_asset_is_under_limit():
    torch.manual_seed(4)
    module = CrossCandidateAttentionModule(input_dim=1024, hidden_dim=128, num_layers=2, num_heads=4)
    weights = state_dict_to_numpy(module)
    assert uncompressed_weight_bytes(weights) < 16 * 1024 * 1024


def test_synthetic_groups_are_learnable():
    rng = np.random.default_rng(5)
    features = rng.normal(scale=0.1, size=(48, 4, 16)).astype(np.float32)
    labels = np.zeros((48, 4), dtype=np.float32)
    for row in range(48):
        correct = row % 4
        features[row, correct, 0] += 4.0
        labels[row, correct] = 1.0
    torch.manual_seed(5)
    module = CrossCandidateAttentionModule(
        input_dim=16, hidden_dim=32, num_layers=2, num_heads=4, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=0.0)
    tensor_x = torch.from_numpy(features)
    tensor_y = torch.from_numpy(labels)
    module.train()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = multi_positive_loss(module(tensor_x), tensor_y, lam=1.0)
        loss.backward()
        optimizer.step()
    module.eval()
    with torch.no_grad():
        chosen = module(tensor_x).argmax(dim=1).numpy()
    hits = labels[np.arange(48), chosen].mean()
    assert hits > 0.9, hits


if __name__ == "__main__":
    test_numpy_matches_torch_eval()
    test_permutation_equivariance()
    test_listwise_loss_matches_hand_calculation()
    test_exported_full_width_asset_is_under_limit()
    test_synthetic_groups_are_learnable()
    print("test_exp_b07 ok")
