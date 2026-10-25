"""EXP-B13: FEAT-G04a centered embeddings as MOD-M03 input.

C1  centered only            vs EXP-B12 arm A
C2  [normalized, centered]   vs EXP-B12 arm A
C3  better of C1/C2 + G01 core + G03   vs EXP-B12 arm B

Three seeds, full models only, single model + POST-P01/P02 as in EXP-B12.
The centered scale is fit on train_all. Test references are not opened.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

import numpy as np  # noqa: E402

from experiments.multiseed import (  # noqa: E402
    RUNS,
    SEEDS,
    ArmInput,
    SplitData,
    configure_determinism,
    decode,
    environment_record,
    reasoning_struct,
    register_seeds,
    rel,
    retention,
    run_arm,
    save_logits,
    summarize,
)
from experiments.exp_b07_cross_attn import HPARAMS, official_score, write_official_predictions  # noqa: E402
from features.centered import centered_embeddings, fit_scale  # noqa: E402


def b12_reference(arm: str) -> Dict[str, Any]:
    audit = json.loads((RUNS / "audit_logs" / "exp_b12_seed_ensemble_audit.json").read_text(encoding="utf-8"))
    per_seed = audit["arms"][arm]["per_seed"]
    return {
        "val_hard": [per_seed[str(s)]["single"]["post"]["val_hard"]["correct"] for s in SEEDS],
        "val_all": [per_seed[str(s)]["single"]["post"]["val_all"]["correct"] for s in SEEDS],
    }


def evaluate(arm: ArmInput, train, val, device, out_dir, registry, ref_name: str) -> Dict[str, Any]:
    print(f"\n--- arm {arm.name} (reference: EXP-B12 {ref_name}) ---")
    run = run_arm(arm, train, val, device, out_dir, with_folds=False)
    registry.extend(run["registry"])
    decoded = {s: decode(run["per_seed"][s]["val_logits_full"], val) for s in SEEDS}
    save_logits(out_dir / f"val_logits_{arm.name}.npz",
                {f"seed_{s}_full": run["per_seed"][s]["val_logits_full"] for s in SEEDS})
    hard = [decoded[s]["post"]["val_hard"]["correct"] for s in SEEDS]
    all_ = [decoded[s]["post"]["val_all"]["correct"] for s in SEEDS]
    ref = b12_reference(ref_name)
    pred = RUNS / "predictions" / f"EXP-B13_{arm.name}_seed42_post_predictions.json"
    write_official_predictions(pred, val.bank, decoded[42]["post_choice"])
    official = official_score(pred, RUNS / "scores" / f"EXP-B13_{arm.name}_seed42_post_score.json")
    if int(official["correct"]) != decoded[42]["post"]["val_all"]["correct"]:
        raise RuntimeError("official scorer disagrees with internal count")
    result = {
        "input_dim": int(arm.dense_train.shape[-1] + (0 if arm.struct_train is None else arm.struct_train.shape[-1])),
        "meta": arm.meta,
        "per_seed": {
            str(s): {
                "full_model": run["per_seed"][s]["full"],
                "decoded": {k: v for k, v in decoded[s].items() if k not in ("argmax_choice", "post_choice")},
            } for s in SEEDS
        },
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
    print(f"{arm.name}: post val_hard {hard} vs ref {ref['val_hard']} | val_all {all_} vs {ref['val_all']} "
          f"| retained {result['retention']['retained']}")
    return result


def run_exp_b13() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B13: centered embeddings (FEAT-G04a)")
    print("=" * 70)
    device = configure_determinism()
    env = environment_record(device)
    train = SplitData("train")
    val = SplitData("validation")

    cen_train = centered_embeddings(train.emb)
    cen_val = centered_embeddings(val.emb)
    scale = fit_scale(cen_train)
    cen_train = (cen_train * scale).astype(np.float32)
    cen_val = (cen_val * scale).astype(np.float32)
    print(f"centered scale (fit on train_all) = {scale:.6f}")
    meta = {"centered_scale": scale, "scale_fit_on": "train_all"}

    out_dir = RUNS / "exp_b13"
    registry = []
    arms: Dict[str, Any] = {}
    c1 = ArmInput("centered_only", cen_train, cen_val, meta=meta)
    c2 = ArmInput("normalized_plus_centered",
                  np.concatenate([train.emb, cen_train], axis=-1), np.concatenate([val.emb, cen_val], axis=-1), meta=meta)
    arms["C1"] = evaluate(c1, train, val, device, out_dir, registry, "embeddings_only")
    arms["C2"] = evaluate(c2, train, val, device, out_dir, registry, "embeddings_only")

    def key(name):
        s = arms[name]["summary"]
        return (s["post_val_hard"]["mean"], s["post_val_all"]["mean"], 1 if name == "C1" else 0)

    picked = max(("C1", "C2"), key=key)
    base = c1 if picked == "C1" else c2
    struct_train, g01_columns, g03_columns = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)
    c3 = ArmInput(f"{base.name}_plus_g01_g03", base.dense_train, base.dense_val, struct_train, struct_val,
                  meta={**meta, "dense_part": picked})
    arms["C3"] = evaluate(c3, train, val, device, out_dir, registry, "embeddings_plus_g01_g03")

    register_seeds("EXP-B13", registry)
    audit = {
        "experiment": "EXP-B13",
        "environment": env,
        "hyperparameters": HPARAMS,
        "seeds": list(SEEDS),
        "centered_scale": scale,
        "c3_dense_part_choice": {"picked": picked, "rule": "higher 3-seed mean post val_hard, then val_all, then C1"},
        "g01_columns": g01_columns,
        "g03_columns": g03_columns,
        "validation_configs_evaluated": sum(a["validation_configs_evaluated"] for a in arms.values()),
        "arms": arms,
        "seed_registry": rel(RUNS / "audit_logs" / "seed_registry.json"),
        "note": "submission.py and model_bundle untouched; test references not opened",
    }
    path = RUNS / "audit_logs" / "exp_b13_centered_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit saved to: {path}")
    return audit


if __name__ == "__main__":
    run_exp_b13()
