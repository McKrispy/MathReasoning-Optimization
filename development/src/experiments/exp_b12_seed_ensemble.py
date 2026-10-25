"""EXP-B12: three seeds, seed ensemble and five-fold ensemble for the B07 / B08 arms.

Arm A  embeddings_only            (EXP-B07 main arm)
Arm B  embeddings_plus_g01_g03    (EXP-B08 specified model)

Hyperparameters are the EXP-B07 constants. Seeds and derived seed streams
come from experiments.multiseed and are written to seed_registry.json.
Validation chooses only the POST-P01/P02 temperature and gamma, as in EXP-B09.
submission.py and model_bundle are not touched. Test references are not opened.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

import numpy as np  # noqa: E402

from experiments.multiseed import (  # noqa: E402
    FOLD_PARTITION_SEED,
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
    save_logits,
    summarize,
)
from experiments.exp_b07_cross_attn import HPARAMS, official_score, write_official_predictions  # noqa: E402

MAC_REFERENCE = {
    "embeddings_only": {"source": "EXP-B07 (Mac MPS, seed 42)", "argmax_val_hard": 294, "argmax_val_all": 730},
    "embeddings_plus_g01_g03": {"source": "EXP-B08 / EXP-B09 (Mac MPS, seed 42)", "argmax_val_hard": 293,
                                "argmax_val_all": 736, "post_val_hard": 294, "post_val_all": 737},
}


def strip(decoded: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in decoded.items() if k not in ("argmax_choice", "post_choice")}


def run_exp_b12() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B12: multi-seed retrain and ensembles")
    print("=" * 70)
    device = configure_determinism()
    env = environment_record(device)
    print(json.dumps(env, indent=2))

    train = SplitData("train")
    val = SplitData("validation")
    tick = time.perf_counter()
    struct_train, g01_columns, g03_columns = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)
    print(f"G01 core {g01_columns}\nG03 {g03_columns}\nfeature extraction {time.perf_counter() - tick:.2f}s")

    arms = [
        ArmInput("embeddings_only", train.emb, val.emb),
        ArmInput("embeddings_plus_g01_g03", train.emb, val.emb, struct_train, struct_val),
    ]
    out_dir = RUNS / "exp_b12"
    results: Dict[str, Any] = {}
    registry = []
    configs_evaluated = 0
    for arm in arms:
        print(f"\n--- arm {arm.name} ---")
        from experiments.multiseed import run_arm
        run = run_arm(arm, train, val, device, out_dir, with_folds=True)
        registry.extend(run["registry"])
        per_seed = run["per_seed"]

        single, folds_ens, logits_dump = {}, {}, {}
        for seed in SEEDS:
            rec = per_seed[seed]
            single[seed] = decode(rec["val_logits_full"], val)
            folds_ens[seed] = decode(rec["val_logits_fold_mean"], val)
            configs_evaluated += single[seed]["validation_configs_evaluated"] + folds_ens[seed]["validation_configs_evaluated"]
            logits_dump[f"seed_{seed}_full"] = rec["val_logits_full"]
            logits_dump[f"seed_{seed}_fold_mean"] = rec["val_logits_fold_mean"]
        seed_ens_logits = np.mean(np.stack([per_seed[s]["val_logits_full"] for s in SEEDS]), axis=0)
        seed_ens = decode(seed_ens_logits, val)
        configs_evaluated += seed_ens["validation_configs_evaluated"]
        logits_dump["seed_ensemble"] = seed_ens_logits
        save_logits(out_dir / f"val_logits_{arm.name}.npz", logits_dump)

        def series(table, key, split):
            return [table[s][key][split]["correct"] for s in SEEDS]

        s_post_h, s_post_a = series(single, "post", "val_hard"), series(single, "post", "val_all")
        f_post_h, f_post_a = series(folds_ens, "post", "val_hard"), series(folds_ens, "post", "val_all")

        pred_path = RUNS / "predictions" / f"EXP-B12_{arm.name}_seed42_post_predictions.json"
        write_official_predictions(pred_path, val.bank, single[42]["post_choice"])
        official = official_score(pred_path, RUNS / "scores" / f"EXP-B12_{arm.name}_seed42_post_score.json")
        if int(official["correct"]) != single[42]["post"]["val_all"]["correct"]:
            raise RuntimeError("official scorer disagrees with internal count")

        results[arm.name] = {
            "per_seed": {
                str(s): {
                    "full_model": per_seed[s]["full"],
                    "folds": per_seed[s]["folds"],
                    "held_out_hard_mean": per_seed[s]["held_out_hard_mean"],
                    "single": strip(single[s]),
                    "fold_ensemble": strip(folds_ens[s]),
                } for s in SEEDS
            },
            "summary": {
                "single_argmax_val_hard": summarize(series(single, "argmax", "val_hard")),
                "single_argmax_val_all": summarize(series(single, "argmax", "val_all")),
                "single_post_val_hard": summarize(s_post_h),
                "single_post_val_all": summarize(s_post_a),
                "fold_ensemble_post_val_hard": summarize(f_post_h),
                "fold_ensemble_post_val_all": summarize(f_post_a),
                "seed_ensemble": strip(seed_ens),
                "held_out_hard_mean_by_seed": summarize([per_seed[s]["held_out_hard_mean"] for s in SEEDS]),
            },
            "retention_vs_single_post": {
                "fold_ensemble": retention(f_post_h, s_post_h, f_post_a, s_post_a, paired=True),
                "seed_ensemble": retention([seed_ens["post"]["val_hard"]["correct"]], s_post_h,
                                           [seed_ens["post"]["val_all"]["correct"]], s_post_a, paired=False),
            },
            "resources": {
                "single_model_bytes": per_seed[42]["full"]["uncompressed_bytes"],
                "seed_ensemble_bytes": sum(per_seed[s]["full"]["uncompressed_bytes"] for s in SEEDS),
                "fold_ensemble_bytes_seed42": sum(r["uncompressed_bytes"] for r in per_seed[42]["folds"]),
            },
            "mac_reference": MAC_REFERENCE[arm.name],
            "official_seed42_post": official,
        }
        summ = results[arm.name]["summary"]
        print(
            f"{arm.name}: single post val_hard {summ['single_post_val_hard']['values']} "
            f"mean {summ['single_post_val_hard']['mean']:.2f}±{summ['single_post_val_hard']['std_ddof1']:.2f} | "
            f"fold-ens {summ['fold_ensemble_post_val_hard']['values']} | "
            f"seed-ens {seed_ens['post']['val_hard']['correct']}"
        )

    register_seeds("EXP-B12", registry)
    audit = {
        "experiment": "EXP-B12",
        "environment": env,
        "hyperparameters": HPARAMS,
        "seeds": list(SEEDS),
        "fold_partition_seed": FOLD_PARTITION_SEED,
        "g01_columns": g01_columns,
        "g03_columns": g03_columns,
        "struct_standardizer_fit_on": "train_all (full models) / fold training rows (fold models)",
        "validation_configs_evaluated": configs_evaluated,
        "arms": results,
        "seed_registry": rel(RUNS / "audit_logs" / "seed_registry.json"),
        "note": "submission.py and model_bundle untouched; test references not opened",
    }
    audit_path = RUNS / "audit_logs" / "exp_b12_seed_ensemble_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b12()
