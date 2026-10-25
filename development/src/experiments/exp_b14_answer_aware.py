"""EXP-B14: MOD-M05 answer-aware attention.

D1  arm A inputs + same-answer matrix S   vs EXP-B12 arm A
D2  arm B inputs + same-answer matrix S   vs EXP-B12 arm B

Three seeds, full models, single model + POST-P01/P02. Candidate-order
augmentation is not run: the model is permutation-equivariant and the loss
sums over candidates, so a permuted copy gives the same loss and gradient.
Test references are not opened.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from experiments.exp_b07_cross_attn import HPARAMS, official_score, write_official_predictions  # noqa: E402
from models.answer_aware_attn import (  # noqa: E402
    AnswerAwareAttentionModule,
    AnswerAwareAttentionNet,
    same_answer_matrix,
)
from models.cross_attn_net import loss_components  # noqa: E402


def run_phase_with_side(model, features, side, labels, epochs, lr, seed, device) -> Dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HPARAMS["weight_decay"])
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(side)),
        torch.from_numpy(np.ascontiguousarray(labels.astype(np.float32))),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=HPARAMS["batch_size"], shuffle=True, generator=generator)
    last = {}
    for _ in range(epochs):
        total, seen = 0.0, 0
        for bx, bs, by in loader:
            bx, bs, by = bx.to(device), bs.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            listwise, bce = loss_components(model(bx, bs), by)
            loss = (listwise + HPARAMS["lambda_bce"] * bce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            optimizer.step()
            total += float(loss.item()) * int(bx.shape[0])
            seen += int(bx.shape[0])
        last = {"loss": total / seen}
    return last


def module_factory(input_dim: int) -> AnswerAwareAttentionModule:
    return AnswerAwareAttentionModule(
        input_dim=input_dim,
        hidden_dim=HPARAMS["hidden_dim"],
        num_layers=HPARAMS["num_layers"],
        num_heads=HPARAMS["num_heads"],
        dropout=HPARAMS["dropout"],
        eps=HPARAMS["layernorm_eps"],
    )


def scorer(weights, features, side):
    return AnswerAwareAttentionNet(weights).forward_numpy(features, side)


def torch_numpy_gap(weights: Dict[str, np.ndarray], features: np.ndarray, side: np.ndarray) -> float:
    module = module_factory(int(features.shape[-1]))
    keys = set(module.state_dict().keys())
    module.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items() if k in keys})
    module.eval()
    with torch.no_grad():
        ref = module(torch.from_numpy(features), torch.from_numpy(side)).numpy()
    return float(np.max(np.abs(ref - scorer(weights, features, side))))


def run_exp_b14() -> Dict[str, Any]:
    from experiments.multiseed import (
        RUNS, SEEDS, WORKSPACE, ArmInput, SplitData, configure_determinism, decode, environment_record,
        reasoning_struct, register_seeds, rel, retention, run_arm, save_logits, summarize,
    )
    from experiments.exp_b13_centered import b12_reference

    print("=" * 70)
    print("Executing EXP-B14: answer-aware attention (MOD-M05)")
    print("=" * 70)
    device = configure_determinism()
    env = environment_record(device)
    train = SplitData("train")
    val = SplitData("validation")
    same_train = same_answer_matrix(train.bank)
    same_val = same_answer_matrix(val.bank)
    print(f"S density train {same_train.mean():.4f} val {same_val.mean():.4f}")
    struct_train, g01_columns, g03_columns = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)

    out_dir = RUNS / "exp_b14"
    registry = []
    arms_out: Dict[str, Any] = {}
    specs = [
        ("D1", ArmInput("embeddings_only_answer_aware", train.emb, val.emb,
                        side_train=same_train, side_val=same_val), "embeddings_only"),
        ("D2", ArmInput("embeddings_plus_g01_g03_answer_aware", train.emb, val.emb, struct_train, struct_val,
                        side_train=same_train, side_val=same_val), "embeddings_plus_g01_g03"),
    ]
    for label, arm, ref_name in specs:
        print(f"\n--- {label} {arm.name} (reference: EXP-B12 {ref_name}) ---")
        run = run_arm(arm, train, val, device, out_dir, with_folds=False,
                      module_factory=module_factory, scorer=scorer)
        registry.extend(run["registry"])
        decoded = {s: decode(run["per_seed"][s]["val_logits_full"], val) for s in SEEDS}
        save_logits(out_dir / f"val_logits_{arm.name}.npz",
                    {f"seed_{s}_full": run["per_seed"][s]["val_logits_full"] for s in SEEDS})
        betas, gaps = {}, {}
        for s in SEEDS:
            with np.load(WORKSPACE / run["per_seed"][s]["full"]["file"]) as w:
                weights = {k: np.array(w[k]) for k in w.files}
            betas[str(s)] = {f"layer{l}": weights[f"blocks.{l}.beta"].tolist() for l in range(HPARAMS["num_layers"])}
            _, x_val, _ = arm.build(np.arange(len(train.bank)))
            gaps[str(s)] = torch_numpy_gap(weights, np.ascontiguousarray(x_val[:32]), same_val[:32])
        if max(gaps.values()) > 1e-4:
            raise RuntimeError(f"NumPy/Torch gap too large: {gaps}")
        hard = [decoded[s]["post"]["val_hard"]["correct"] for s in SEEDS]
        all_ = [decoded[s]["post"]["val_all"]["correct"] for s in SEEDS]
        ref = b12_reference(ref_name)
        pred = RUNS / "predictions" / f"EXP-B14_{arm.name}_seed42_post_predictions.json"
        write_official_predictions(pred, val.bank, decoded[42]["post_choice"])
        official = official_score(pred, RUNS / "scores" / f"EXP-B14_{arm.name}_seed42_post_score.json")
        if int(official["correct"]) != decoded[42]["post"]["val_all"]["correct"]:
            raise RuntimeError("official scorer disagrees with internal count")
        arms_out[label] = {
            "arm": arm.name,
            "input_dim": int(arm.dense_train.shape[-1] + (0 if arm.struct_train is None else arm.struct_train.shape[-1])),
            "per_seed": {
                str(s): {
                    "full_model": run["per_seed"][s]["full"],
                    "decoded": {k: v for k, v in decoded[s].items() if k not in ("argmax_choice", "post_choice")},
                } for s in SEEDS
            },
            "betas": betas,
            "numpy_torch_max_abs": gaps,
            "summary": {
                "argmax_val_hard": summarize([decoded[s]["argmax"]["val_hard"]["correct"] for s in SEEDS]),
                "argmax_val_all": summarize([decoded[s]["argmax"]["val_all"]["correct"] for s in SEEDS]),
                "post_val_hard": summarize(hard),
                "post_val_all": summarize(all_),
            },
            "reference": {"arm": f"EXP-B12 {ref_name}", **ref},
            "retention": retention(hard, ref["val_hard"], all_, ref["val_all"], paired=True),
            "official_seed42_post": official,
            "validation_configs_evaluated": sum(decoded[s]["validation_configs_evaluated"] for s in SEEDS),
        }
        print(f"{label}: post val_hard {hard} vs ref {ref['val_hard']} | val_all {all_} vs {ref['val_all']} "
              f"| retained {arms_out[label]['retention']['retained']} | betas {betas}")

    register_seeds("EXP-B14", registry)
    audit = {
        "experiment": "EXP-B14",
        "environment": env,
        "hyperparameters": HPARAMS,
        "seeds": list(SEEDS),
        "same_answer_density": {"train": float(same_train.mean()), "validation": float(same_val.mean())},
        "g01_columns": g01_columns,
        "g03_columns": g03_columns,
        "augmentation": "not run: permutation-equivariant model and permutation-invariant loss make it a no-op",
        "validation_configs_evaluated": sum(a["validation_configs_evaluated"] for a in arms_out.values()),
        "arms": arms_out,
        "seed_registry": rel(RUNS / "audit_logs" / "seed_registry.json"),
        "note": "submission.py and model_bundle untouched; test references not opened",
    }
    path = RUNS / "audit_logs" / "exp_b14_answer_aware_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit saved to: {path}")
    return audit


if __name__ == "__main__":
    run_exp_b14()
