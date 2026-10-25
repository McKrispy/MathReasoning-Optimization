"""EXP-B11: Family-level permutation of the trained EXP-B10 tree.

The tree is not refit. On val_hard, one family at a time has its question
blocks reassigned to other val_hard questions. The drop in selection accuracy
is the dependence measure. Split gain is not used. Test references are not opened.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import build_label_matrix, get_hard_subset_mask, load_dataset, load_gold_references  # noqa: E402
from experiments.exp_b07_cross_attn import accuracy_of, official_score, write_official_predictions  # noqa: E402
from experiments.exp_b08_reasoning import load_contract, select_reasoning_columns  # noqa: E402
from experiments.exp_b09_calibration import hits_of, structured_features  # noqa: E402

N_REPEATS = 20
SEED = 42


def family_column_index(columns: Sequence[str], contract: Dict[str, Any]) -> Dict[str, List[int]]:
    """Map each saved column onto the contract family that lists it."""
    owner = {}
    for family, members in contract["statistically_eligible_by_family"].items():
        for name in members:
            owner[name] = family
    grouped: Dict[str, List[int]] = {}
    missing = []
    for index, name in enumerate(columns):
        family = owner.get(name)
        if family is None:
            missing.append(name)
            continue
        grouped.setdefault(family, []).append(index)
    if missing:
        raise RuntimeError(f"B10 columns are not in the contract families: {missing}")
    return grouped


def shuffle_question_blocks(features: np.ndarray, column_index: Sequence[int], permutation: np.ndarray) -> np.ndarray:
    """Move one family's [4, D] block between questions. Other columns stay put."""
    shuffled = np.array(features, copy=True)
    shuffled[:, :, list(column_index)] = features[permutation][:, :, list(column_index)]
    return shuffled


def predict_choice(tree, features: np.ndarray) -> np.ndarray:
    flat = features.reshape(features.shape[0] * features.shape[1], features.shape[2])
    probabilities = tree.predict_proba(flat)[:, 1].reshape(features.shape[0], features.shape[1])
    return np.argmax(probabilities, axis=1).astype(int)


def run_exp_b11() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B11: Family shuffle of the trained tree")
    print("=" * 70)
    runs = WORKSPACE / "dev" / "runs"
    config = json.loads((runs / "exp_b10" / "config.json").read_text(encoding="utf-8"))
    columns = list(config["m04"]["columns"])
    contract = load_contract()
    families = family_column_index(columns, contract)
    g01_columns, g03_columns = select_reasoning_columns(contract)
    print(f"Families: { {name: len(index) for name, index in families.items()} }")

    with (runs / "exp_b10" / "gbdt.pkl").open("rb") as handle:
        tree = pickle.load(handle)

    bank_val, arrays_val = load_dataset("validation")
    labels_val = build_label_matrix(bank_val, load_gold_references("validation"))
    hard_val = get_hard_subset_mask(bank_val)
    features_val = structured_features(bank_val, arrays_val, g01_columns, g03_columns)
    built_names = list(g01_columns) + list(g03_columns)
    if built_names != columns:
        order = [built_names.index(name) for name in columns]
        features_val = features_val[:, :, order]

    features_hard = features_val[hard_val]
    labels_hard = labels_val[hard_val]
    baseline_choice = predict_choice(tree, features_hard)
    baseline_hits = hits_of(baseline_choice, labels_hard)
    baseline = accuracy_of(baseline_hits)
    full_choice = predict_choice(tree, features_val)
    full_hits = hits_of(full_choice, labels_val)
    baseline_all = accuracy_of(full_hits)
    b10_audit = json.loads((runs / "audit_logs" / "exp_b10_fusion_audit.json").read_text(encoding="utf-8"))
    expected = b10_audit["m04_reference"]["val_hard"]["correct"]
    if baseline["correct"] != expected:
        raise RuntimeError(f"reloaded tree val_hard {baseline['correct']} != B10 record {expected}")
    print(f"Baseline val_hard {baseline['correct']}/{baseline['total']}")

    pred_path = runs / "predictions" / "EXP-B11_tree_baseline_predictions.json"
    score_path = runs / "scores" / "EXP-B11_tree_baseline_score.json"
    write_official_predictions(pred_path, bank_val, full_choice)
    official = official_score(pred_path, score_path)
    if int(official["correct"]) != baseline_all["correct"]:
        raise RuntimeError("official score disagrees with the reloaded tree")

    rng = np.random.default_rng(SEED)
    family_rows = []
    tick = time.perf_counter()
    for family, column_index in families.items():
        repeat_correct = []
        for _ in range(N_REPEATS):
            permutation = rng.permutation(features_hard.shape[0])
            shuffled = shuffle_question_blocks(features_hard, column_index, permutation)
            chosen = predict_choice(tree, shuffled)
            repeat_correct.append(int(hits_of(chosen, labels_hard).sum()))
        correct = np.asarray(repeat_correct, dtype=np.float64)
        mean_correct = float(correct.mean())
        row = {
            "family": family,
            "n_columns": len(column_index),
            "columns": [columns[index] for index in column_index],
            "repeats": N_REPEATS,
            "baseline_correct": baseline["correct"],
            "mean_correct": mean_correct,
            "std_correct": float(correct.std()),
            "min_correct": int(correct.min()),
            "max_correct": int(correct.max()),
            "mean_drop_questions": baseline["correct"] - mean_correct,
            "mean_accuracy": mean_correct / baseline["total"],
            "mean_drop_accuracy": (baseline["correct"] - mean_correct) / baseline["total"],
        }
        family_rows.append(row)
        print(
            f"  {family}: mean {mean_correct:.2f}/{baseline['total']} "
            f"drop {row['mean_drop_questions']:.2f} questions"
        )
    elapsed = time.perf_counter() - tick
    family_rows.sort(key=lambda row: row["mean_drop_questions"], reverse=True)

    audit = {
        "experiment": "EXP-B11",
        "model": "dev/runs/exp_b10/gbdt.pkl",
        "rows": "val_hard",
        "shuffle": "question blocks within val_hard; one family at a time; other columns unchanged",
        "repeats": N_REPEATS,
        "seed": SEED,
        "split_gain_used": False,
        "baseline_val_hard": baseline,
        "baseline_val_all": baseline_all,
        "official_validation_score": official,
        "families": family_rows,
        "shuffle_seconds": elapsed,
        "files": {
            "predictions": "dev/runs/predictions/EXP-B11_tree_baseline_predictions.json",
            "score": "dev/runs/scores/EXP-B11_tree_baseline_score.json",
        },
    }
    audit_path = runs / "audit_logs" / "exp_b11_family_shuffle_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b11()
