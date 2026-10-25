"""EXP-B10: MOD-M03 plus a tree reranker and disagreement gating (RQ5).

MOD-M03 is the frozen EXP-B09 policy. MOD-M04 is a histogram GBDT fit only on
train_hard, using the B04 retained columns and the EXP-B05 tree settings.
Validation chooses the gate. It does not choose tree hyperparameters.
Test references are not opened.
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
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import build_label_matrix, get_hard_subset_mask, load_dataset, load_gold_references  # noqa: E402
from models.cross_attn_net import CrossCandidateAttentionNet, uncompressed_weight_bytes  # noqa: E402
from postprocessing.aggregation import WeightedSelfConsistencyAggregator, apply_disagreement_gate  # noqa: E402
from postprocessing.calibration import TemperatureScaler  # noqa: E402
from experiments.exp_b07_cross_attn import accuracy_of, official_score, write_official_predictions  # noqa: E402
from experiments.exp_b08_reasoning import load_contract, select_reasoning_columns  # noqa: E402
from experiments.exp_b09_calibration import build_model_input, hits_of, load_npz, structured_features  # noqa: E402

TREE_PARAMS = {
    "max_iter": 100,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}
CONFIDENCE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def gate_specs() -> List[Dict[str, Any]]:
    specs = [
        {"name": "m03_only", "threshold": None, "priority": 0},
        {"name": "m04_only", "threshold": None, "priority": 1},
        {"name": "higher_confidence", "threshold": None, "priority": 2},
    ]
    for offset, threshold in enumerate(CONFIDENCE_THRESHOLDS):
        specs.append({
            "name": "m04_when_m03_below",
            "threshold": threshold,
            "priority": 3 + offset,
        })
    return specs


def select_gate(rows: Sequence[Dict[str, Any]], m03_val_all_correct: int) -> Dict[str, Any]:
    """Highest val_hard among gates that do not lower val_all below MOD-M03.

    Ties keep the higher val_all, then the earlier priority. m03_only is first.
    """
    eligible = [row for row in rows if row["val_all"]["correct"] >= m03_val_all_correct]
    pool = eligible if eligible else list(rows)
    return max(
        pool,
        key=lambda row: (row["val_hard"]["correct"], row["val_all"]["correct"], -row["priority"]),
    )


def score_choices(chosen: np.ndarray, labels: np.ndarray, hard_mask: np.ndarray) -> Dict[str, Any]:
    hits = hits_of(chosen, labels)
    return {"val_hard": accuracy_of(hits[hard_mask]), "val_all": accuracy_of(hits)}


def run_exp_b10() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B10: Tree fusion and disagreement gating (RQ5)")
    print("=" * 70)
    runs = WORKSPACE / "dev" / "runs"
    b09_config = json.loads((runs / "exp_b09" / "config.json").read_text(encoding="utf-8"))
    g01_columns, g03_columns = select_reasoning_columns(load_contract())
    if g01_columns != b09_config["g01_columns"] or g03_columns != b09_config["g03_columns"]:
        raise RuntimeError("B09 column list does not match the current contract")
    columns = list(g01_columns) + list(g03_columns)

    bank_train, arrays_train = load_dataset("train")
    labels_train = build_label_matrix(bank_train, load_gold_references("train"))
    hard_train = get_hard_subset_mask(bank_train)
    print("Extracting training features...")
    tick = time.perf_counter()
    struct_train = structured_features(bank_train, arrays_train, g01_columns, g03_columns)
    train_feature_seconds = time.perf_counter() - tick
    x_hard = struct_train[hard_train].reshape(int(hard_train.sum()) * 4, len(columns))
    y_hard = labels_train[hard_train].reshape(-1)

    tree = HistGradientBoostingClassifier(**TREE_PARAMS)
    tick = time.perf_counter()
    tree.fit(x_hard, y_hard)
    fit_seconds = time.perf_counter() - tick
    print(f"GBDT fit on {int(hard_train.sum())} hard questions in {fit_seconds:.2f}s")

    bank_val, arrays_val = load_dataset("validation")
    labels_val = build_label_matrix(bank_val, load_gold_references("validation"))
    hard_val = get_hard_subset_mask(bank_val)

    tick = time.perf_counter()
    struct_val = structured_features(bank_val, arrays_val, g01_columns, g03_columns)
    feature_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    m04_probs = tree.predict_proba(struct_val.reshape(len(bank_val) * 4, len(columns)))[:, 1]
    m04_probs = m04_probs.reshape(len(bank_val), 4).astype(np.float32)
    m04_seconds = time.perf_counter() - tick
    m04_choice = np.argmax(m04_probs, axis=1).astype(int)
    m04_conf = m04_probs[np.arange(len(bank_val)), m04_choice]

    weights = load_npz(runs / "exp_b08" / "arm_embeddings_plus_g01_g03.npz")
    features_val = build_model_input(
        bank_val, arrays_val, weights["struct_mean"], weights["struct_std"], g01_columns, g03_columns,
    )
    tick = time.perf_counter()
    logits = CrossCandidateAttentionNet(weights).forward_numpy(features_val)
    m03_probs = TemperatureScaler(b09_config["temperature"]).transform(logits)
    m03_choice = np.asarray(
        WeightedSelfConsistencyAggregator(gamma=float(b09_config["gamma"])).aggregate(bank_val, m03_probs),
        dtype=int,
    )
    m03_seconds = time.perf_counter() - tick
    m03_conf = m03_probs[np.arange(len(bank_val)), m03_choice]

    b09_payload = json.loads((runs / "predictions" / "EXP-B09_calibrated_predictions.json").read_text(encoding="utf-8"))
    b09_by_id = dict(zip(b09_payload["ids"], b09_payload["selected"]))
    b09_choice = np.asarray([int(b09_by_id[row["id"]]) for row in bank_val], dtype=int)
    if not np.array_equal(m03_choice, b09_choice):
        raise RuntimeError("recomputed MOD-M03 choices do not match EXP-B09")

    rows = []
    for spec in gate_specs():
        chosen = apply_disagreement_gate(
            m03_choice, m04_choice, m03_conf, m04_conf, spec["name"], spec["threshold"],
        )
        scored = score_choices(chosen, labels_val, hard_val)
        row = {
            "name": spec["name"],
            "threshold": spec["threshold"],
            "priority": spec["priority"],
            **scored,
            "choice_histogram_val_all": np.bincount(chosen, minlength=4).tolist(),
        }
        rows.append((row, chosen))
        print(
            f"  {spec['name']} t={spec['threshold']}: "
            f"val_hard {scored['val_hard']['correct']}/{scored['val_hard']['total']} "
            f"val_all {scored['val_all']['correct']}/{scored['val_all']['total']}"
        )

    m03_correct = rows[0][0]["val_all"]["correct"]
    selected_row = select_gate([row for row, _ in rows], m03_correct)
    selected_choice = next(
        choice for row, choice in rows
        if row["name"] == selected_row["name"] and row["threshold"] == selected_row["threshold"]
    )

    disagree = m03_choice != m04_choice
    m03_hits = hits_of(m03_choice, labels_val)
    m04_hits = hits_of(m04_choice, labels_val)
    hard_disagree = disagree & hard_val

    out_dir = runs / "exp_b10"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "gbdt.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(tree, handle)
    model_bytes = model_path.stat().st_size
    base_bytes = uncompressed_weight_bytes(weights)
    b09_config_bytes = (runs / "exp_b09" / "config.json").stat().st_size

    pred_path = runs / "predictions" / "EXP-B10_fusion_predictions.json"
    score_path = runs / "scores" / "EXP-B10_fusion_score.json"
    write_official_predictions(pred_path, bank_val, selected_choice)
    official = official_score(pred_path, score_path)
    if int(official["correct"]) != int(hits_of(selected_choice, labels_val).sum()):
        raise RuntimeError("official score disagrees with internal val_all")

    config = {
        "m03": {
            "weights": b09_config["base_weights"],
            "temperature": b09_config["temperature"],
            "gamma": b09_config["gamma"],
        },
        "m04": {
            "train_rows": "train_hard",
            "columns": columns,
            "tree_params": TREE_PARAMS,
            "sklearn": sklearn.__version__,
        },
        "selected_gate": {
            "name": selected_row["name"],
            "threshold": selected_row["threshold"],
        },
        "selection_rule": "max val_hard among gates whose val_all is at least MOD-M03; ties keep higher val_all, then lower priority, and m03_only has priority 0",
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_bytes = config_path.stat().st_size

    audit = {
        "experiment": "EXP-B10",
        "research_question": "RQ5",
        "sklearn": sklearn.__version__,
        "columns": columns,
        "tree_params": TREE_PARAMS,
        "train_questions": int(hard_train.sum()),
        "fit_seconds": fit_seconds,
        "train_feature_seconds": train_feature_seconds,
        "m03_reference": score_choices(m03_choice, labels_val, hard_val),
        "m04_reference": score_choices(m04_choice, labels_val, hard_val),
        "gates": [row for row, _ in rows],
        "selected": selected_row,
        "official_validation_score": official,
        "disagreement_val_hard": {
            "questions": int(hard_disagree.sum()),
            "m03_correct": int(m03_hits[hard_disagree].sum()),
            "m04_correct": int(m04_hits[hard_disagree].sum()),
            "both_correct": int((m03_hits & m04_hits)[hard_disagree].sum()),
            "both_wrong": int((~m03_hits & ~m04_hits)[hard_disagree].sum()),
            "only_m03": int((m03_hits & ~m04_hits)[hard_disagree].sum()),
            "only_m04": int((~m03_hits & m04_hits)[hard_disagree].sum()),
        },
        "resources": {
            "feature_seconds_val_all": feature_seconds,
            "m03_seconds_val_all": m03_seconds,
            "m04_seconds_val_all": m04_seconds,
            "combined_seconds_val_all": feature_seconds + m03_seconds + m04_seconds,
            "gbdt_bytes": model_bytes,
            "m03_uncompressed_bytes": base_bytes,
            "b09_config_bytes": b09_config_bytes,
            "b10_config_bytes": config_bytes,
            "fused_uncompressed_bytes": base_bytes + b09_config_bytes + model_bytes + config_bytes,
        },
        "files": {
            "gbdt": "dev/runs/exp_b10/gbdt.pkl",
            "config": "dev/runs/exp_b10/config.json",
            "predictions": "dev/runs/predictions/EXP-B10_fusion_predictions.json",
            "score": "dev/runs/scores/EXP-B10_fusion_score.json",
        },
    }
    audit_path = runs / "audit_logs" / "exp_b10_fusion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {selected_row['name']} t={selected_row['threshold']} "
        f"val_hard {selected_row['val_hard']['correct']}/{selected_row['val_hard']['total']} "
        f"val_all {selected_row['val_all']['correct']}/{selected_row['val_all']['total']}"
    )
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b10()
