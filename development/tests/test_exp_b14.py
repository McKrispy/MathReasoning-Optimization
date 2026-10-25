"""Fast checks for MOD-M05. The full EXP-B14 fit is the experiment script, not this test."""
from pathlib import Path
import sys
import numpy as np
import torch

DEV_SRC = Path(__file__).resolve().parents[1] / "src"
if str(DEV_SRC) not in sys.path:
    sys.path.insert(0, str(DEV_SRC))

from models.answer_aware_attn import (  # noqa: E402
    AnswerAwareAttentionModule,
    AnswerAwareAttentionNet,
    same_answer_matrix,
)
from models.cross_attn_net import CrossCandidateAttentionModule, CrossCandidateAttentionNet, state_dict_to_numpy  # noqa: E402


def _toy(seed: int = 0, beta: float = 0.0) -> AnswerAwareAttentionModule:
    torch.manual_seed(seed)
    module = AnswerAwareAttentionModule(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4, dropout=0.1)
    with torch.no_grad():
        for block in module.blocks:
            block.beta.fill_(beta)
    module.eval()
    return module


def _inputs(seed: int = 1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 4, 16)).astype(np.float32)
    same = np.zeros((6, 4, 4), dtype=np.float32)
    same[:, 0, 1] = same[:, 1, 0] = 1.0
    same[:3, 2, 3] = same[:3, 3, 2] = 1.0
    return x, same


def test_numpy_matches_torch():
    module = _toy(beta=0.7)
    x, same = _inputs()
    with torch.no_grad():
        ref = module(torch.from_numpy(x), torch.from_numpy(same)).numpy()
    got = AnswerAwareAttentionNet(state_dict_to_numpy(module)).forward_numpy(x, same)
    assert float(np.max(np.abs(ref - got))) < 1e-5


def test_permutation_equivariance_with_same_matrix():
    module = _toy(seed=3, beta=1.3)
    x, same = _inputs(4)
    perm = np.array([2, 0, 3, 1])
    weights = state_dict_to_numpy(module)
    base = AnswerAwareAttentionNet(weights).forward_numpy(x, same)
    permuted = AnswerAwareAttentionNet(weights).forward_numpy(x[:, perm], same[:, perm][:, :, perm])
    assert np.allclose(permuted, base[:, perm], atol=1e-5)


def test_zero_beta_equals_mod_m03_with_same_seed():
    aware = _toy(seed=5, beta=0.0)
    torch.manual_seed(5)
    plain = CrossCandidateAttentionModule(input_dim=16, hidden_dim=32, num_layers=2, num_heads=4, dropout=0.1)
    plain.eval()
    x, same = _inputs(6)
    a = AnswerAwareAttentionNet(state_dict_to_numpy(aware)).forward_numpy(x, same)
    b = CrossCandidateAttentionNet(state_dict_to_numpy(plain)).forward_numpy(x)
    assert np.allclose(a, b, atol=1e-6)


def test_same_answer_matrix_marks_only_matching_valid_pairs():
    bank = [{"candidates": [
        {"text": "2 + 3 = 5. \\boxed{5}"}, {"text": "So the answer is \\boxed{5.0}"},
        {"text": "\\boxed{7}"}, {"text": "no final answer"},
    ]}]
    same = same_answer_matrix(bank)[0]
    assert same[0, 1] == 1 and same[1, 0] == 1
    assert same.trace() == 0
    assert same[2].sum() == 0 and same[3].sum() == 0
