"""EXP-B15 R5 follow-up: family shuffle of the retained tree (EXP-B11 protocol).

Loads the seed 42 M04a_train_all_plus_svd tree, rebuilds its 144 input columns
(G01 core 4 + G03 12 + FEAT-G04b 128) and moves one family's [4, D] block between
val_hard questions at a time, 20 repeats, rng seed 42, argmax decoding.
"""
import json
import pickle
import sys
from pathlib import Path

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

import numpy as np  # noqa: E402

from experiments.exp_b11_family_shuffle import N_REPEATS, SEED, shuffle_question_blocks  # noqa: E402
from experiments.exp_b15_tree import svd_features, tree_logits  # noqa: E402
from experiments.multiseed import RUNS, SplitData, reasoning_struct, rel  # noqa: E402

TREE_NAME = "M04a_train_all_plus_svd"


def run() -> dict:
    train = SplitData("train")
    val = SplitData("validation")
    struct_train, g01_columns, g03_columns = reasoning_struct(train)
    struct_val, _, _ = reasoning_struct(val)
    _, svd_val, _ = svd_features(train, val)
    x_val = np.concatenate([struct_val, svd_val], axis=-1)
    path = RUNS / "exp_b15" / "seed_42" / f"{TREE_NAME}.pkl"
    with path.open("rb") as handle:
        tree = pickle.load(handle)

    n_g01, n_g03 = len(g01_columns), len(g03_columns)
    families = {
        "G01_core": list(range(n_g01)),
        "G03": list(range(n_g01, n_g01 + n_g03)),
        "G04b_svd": list(range(n_g01 + n_g03, x_val.shape[-1])),
    }
    x_hard, y_hard = x_val[val.hard], val.labels[val.hard]
    idx = np.arange(len(x_hard))

    def correct(features: np.ndarray) -> int:
        choice = np.argmax(tree_logits("M04a", tree, features), axis=1)
        return int(y_hard[idx, choice].astype(bool).sum())

    baseline = correct(x_hard)
    audit = json.loads((RUNS / "audit_logs" / "exp_b15_tree_audit.json").read_text(encoding="utf-8"))
    expected = int(audit["r5"]["per_seed"]["42"]["decoded"]["argmax"]["val_hard"]["correct"])
    if baseline != expected:
        raise RuntimeError(f"reloaded tree val_hard {baseline} != EXP-B15 record {expected}")
    print(f"Baseline val_hard {baseline}/{len(x_hard)}")

    rng = np.random.default_rng(SEED)
    rows = []
    for family, columns in families.items():
        scores = np.asarray([correct(shuffle_question_blocks(x_hard, columns, rng.permutation(len(x_hard))))
                             for _ in range(N_REPEATS)], dtype=np.float64)
        row = {"family": family, "n_columns": len(columns), "repeats": N_REPEATS, "baseline_correct": baseline,
               "mean_correct": float(scores.mean()), "std_correct": float(scores.std()),
               "min_correct": int(scores.min()), "max_correct": int(scores.max()),
               "mean_drop_questions": baseline - float(scores.mean())}
        rows.append(row)
        print(f"  {family} ({len(columns)} cols): mean {row['mean_correct']:.2f} drop {row['mean_drop_questions']:.2f}")

    result = {
        "experiment": "EXP-B15 R5 family shuffle",
        "tree": rel(path),
        "shuffle": "question blocks within val_hard; one family at a time; other columns unchanged; argmax",
        "rng_seed": SEED,
        "families": rows,
        "note": "trees are seed-invariant in EXP-B15, so seed 42 stands for all three",
    }
    out = RUNS / "audit_logs" / "exp_b15_shuffle_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit saved to: {out}")
    return result


if __name__ == "__main__":
    run()
