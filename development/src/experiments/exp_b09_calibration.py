"""EXP-B09: Temperature scaling and weighted answer-cluster aggregation (RQ4).

The frozen MOD-M03 is the EXP-B08 model: 1024-d embeddings plus the B04
retained columns (G01 consensus core and eligible G03). This batch does not
refit those weights. Validation labels choose only the postprocessing:

  T    minimizes candidate-level binary NLL. It is not chosen by accuracy.
  gamma  is scanned on the fixed grid below. The retained setting is the
         highest val_hard accuracy whose val_all does not fall below argmax.
         Ties keep the higher val_all, then the smaller gamma.

Test references are not opened.
"""
import json
import os
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
from features.consensus import ConsensusFeatureExtractor  # noqa: E402
from features.symbolic import SymbolicVerificationFeatureExtractor  # noqa: E402
from models.cross_attn_net import CrossCandidateAttentionNet, uncompressed_weight_bytes  # noqa: E402
from postprocessing.aggregation import WeightedSelfConsistencyAggregator  # noqa: E402
from postprocessing.calibration import (  # noqa: E402
    TemperatureScaler,
    binary_nll,
    expected_calibration_error,
    sigmoid_probs,
)
from experiments.exp_b07_cross_attn import (  # noqa: E402
    accuracy_of,
    l2_normalize,
    official_score,
    standardize_apply,
    write_official_predictions,
)
from experiments.exp_b08_reasoning import (  # noqa: E402
    load_contract,
    select_reasoning_columns,
    take_columns,
)

GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def structured_features(bank, arrays, g01_columns: Sequence[str], g03_columns: Sequence[str]) -> np.ndarray:
    g01_names = ConsensusFeatureExtractor.FEATURE_NAMES
    g03_names = SymbolicVerificationFeatureExtractor.FEATURE_NAMES
    g01 = take_columns(ConsensusFeatureExtractor().transform(bank, arrays), g01_names, g01_columns)
    g03 = take_columns(SymbolicVerificationFeatureExtractor().transform(bank, arrays), g03_names, g03_columns)
    return np.concatenate([g01, g03], axis=-1).astype(np.float32)


def build_model_input(bank, arrays, mean: np.ndarray, std: np.ndarray, g01_columns, g03_columns) -> np.ndarray:
    embeddings = l2_normalize(arrays["embeddings"])
    structured = standardize_apply(structured_features(bank, arrays, g01_columns, g03_columns), mean, std)
    return np.concatenate([embeddings, structured], axis=-1).astype(np.float32)


def hits_of(chosen: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return labels[np.arange(labels.shape[0]), chosen].astype(bool)


def choice_record(chosen: np.ndarray, labels: np.ndarray, hard_mask: np.ndarray, gamma, temperature: float) -> Dict[str, Any]:
    hits = hits_of(chosen, labels)
    return {
        "gamma": gamma,
        "temperature": temperature,
        "val_hard": accuracy_of(hits[hard_mask]),
        "val_all": accuracy_of(hits),
        "choice_histogram_val_all": np.bincount(chosen, minlength=4).tolist(),
    }


def select_gamma(rows: Sequence[Dict[str, Any]], argmax_val_all_correct: int) -> Dict[str, Any]:
    """Highest val_hard among settings that do not lower val_all. Then smaller gamma."""
    eligible = [row for row in rows if row["val_all"]["correct"] >= argmax_val_all_correct]
    pool = eligible if eligible else list(rows)
    return max(
        pool,
        key=lambda row: (row["val_hard"]["correct"], row["val_all"]["correct"], -float(row["gamma"])),
    )


def quartile_reliability(probabilities: np.ndarray, hits: np.ndarray) -> List[Dict[str, Any]]:
    if probabilities.size == 0:
        return []
    edges = np.quantile(probabilities, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for bin_id in range(4):
        low, high = float(edges[bin_id]), float(edges[bin_id + 1])
        if bin_id == 3:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        rows.append({
            "quartile": bin_id,
            "low": low,
            "high": high,
            "count": count,
            "mean_confidence": float(probabilities[mask].mean()) if count else None,
            "accuracy": float(hits[mask].mean()) if count else None,
            "correct": int(hits[mask].sum()) if count else 0,
        })
    return rows


def disagreement(reference: np.ndarray, new: np.ndarray, labels: np.ndarray, hard_mask: np.ndarray) -> Dict[str, Any]:
    differ = (reference != new) & hard_mask
    ref_hits = hits_of(reference, labels)
    new_hits = hits_of(new, labels)
    return {
        "val_hard_disagreements": int(differ.sum()),
        "reference_correct_on_disagreement": int(ref_hits[differ].sum()),
        "new_correct_on_disagreement": int(new_hits[differ].sum()),
        "reference_accuracy_on_disagreement": accuracy_of(ref_hits[differ]),
        "new_accuracy_on_disagreement": accuracy_of(new_hits[differ]),
    }


def run_exp_b09() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B09: Calibration and weighted aggregation (RQ4)")
    print("=" * 70)
    runs = WORKSPACE / "dev" / "runs"
    weight_path = runs / "exp_b08" / "arm_embeddings_plus_g01_g03.npz"
    weights = load_npz(weight_path)
    g01_columns, g03_columns = select_reasoning_columns(load_contract())
    if int(weights["struct_column_count"][0]) != len(g01_columns) + len(g03_columns):
        raise RuntimeError("B08 checkpoint column count does not match the B04 retained families")

    bank_val, arrays_val = load_dataset("validation")
    labels_val = build_label_matrix(bank_val, load_gold_references("validation"))
    hard_val = get_hard_subset_mask(bank_val)
    print(f"Rebuilding validation features for {len(bank_val)} questions...")
    features_val = build_model_input(
        bank_val, arrays_val, weights["struct_mean"], weights["struct_std"], g01_columns, g03_columns,
    )
    net = CrossCandidateAttentionNet(weights)
    tick = time.perf_counter()
    logits = net.forward_numpy(features_val)
    forward_seconds = time.perf_counter() - tick
    argmax_choice = np.argmax(logits, axis=1).astype(int)

    b08_path = runs / "predictions" / "EXP-B08_reasoning_predictions.json"
    b08_payload = json.loads(b08_path.read_text(encoding="utf-8"))
    b08_by_id = dict(zip(b08_payload["ids"], b08_payload["selected"]))
    b08_choice = np.asarray([int(b08_by_id[row["id"]]) for row in bank_val], dtype=int)
    if not np.array_equal(argmax_choice, b08_choice):
        raise RuntimeError("reloaded B08 argmax does not reproduce EXP-B08 predictions")
    print("Reloaded B08 argmax matches the saved predictions.")

    b01_path = runs / "predictions" / "EXP-B01_validation_predictions.json"
    b01_payload = json.loads(b01_path.read_text(encoding="utf-8"))
    b01_by_id = dict(zip(b01_payload["ids"], b01_payload["selected"]))
    b01_choice = np.asarray([int(b01_by_id[row["id"]]) for row in bank_val], dtype=int)

    scaler = TemperatureScaler().fit(logits, labels_val)
    temperature = scaler.temperature
    raw_probs = sigmoid_probs(logits, 1.0).astype(np.float32)
    calibrated_probs = scaler.transform(logits)
    nll_before = binary_nll(raw_probs, labels_val)
    nll_after = binary_nll(calibrated_probs, labels_val)
    ece_before, bins_before = expected_calibration_error(raw_probs, labels_val)
    ece_after, bins_after = expected_calibration_error(calibrated_probs, labels_val)
    hard_candidate = np.repeat(hard_val, 4)
    ece_hard_before, _ = expected_calibration_error(raw_probs.reshape(-1)[hard_candidate], labels_val.reshape(-1)[hard_candidate])
    ece_hard_after, _ = expected_calibration_error(calibrated_probs.reshape(-1)[hard_candidate], labels_val.reshape(-1)[hard_candidate])
    print(f"Temperature T={temperature:.6f}  NLL {nll_before:.6f} -> {nll_after:.6f}  ECE {ece_before:.6f} -> {ece_after:.6f}")

    argmax_record = choice_record(argmax_choice, labels_val, hard_val, gamma=None, temperature=1.0)
    sweeps = {"temperature_1": [], "calibrated": []}
    tick = time.perf_counter()
    for gamma in GAMMA_GRID:
        for key, probs, used_t in (
            ("temperature_1", raw_probs, 1.0),
            ("calibrated", calibrated_probs, temperature),
        ):
            chosen = np.asarray(
                WeightedSelfConsistencyAggregator(gamma=gamma).aggregate(bank_val, probs),
                dtype=int,
            )
            row = choice_record(chosen, labels_val, hard_val, gamma, used_t)
            row["chosen"] = chosen
            sweeps[key].append(row)
            print(
                f"  T={used_t:.4f} gamma={gamma:<4} "
                f"val_hard {row['val_hard']['correct']}/{row['val_hard']['total']} "
                f"val_all {row['val_all']['correct']}/{row['val_all']['total']}"
            )
    sweep_seconds = time.perf_counter() - tick

    selected = select_gamma(sweeps["calibrated"], argmax_record["val_all"]["correct"])
    selected_choice = selected.pop("chosen")
    for row in sweeps["temperature_1"] + sweeps["calibrated"]:
        row.pop("chosen", None)

    pred_path = runs / "predictions" / "EXP-B09_calibrated_predictions.json"
    score_path = runs / "scores" / "EXP-B09_calibrated_score.json"
    write_official_predictions(pred_path, bank_val, selected_choice)
    official = official_score(pred_path, score_path)
    if int(official["correct"]) != int(np.sum(hits_of(selected_choice, labels_val))):
        raise RuntimeError("official score disagrees with internal val_all")

    argmax_hits = hits_of(argmax_choice, labels_val)
    selected_hits = hits_of(selected_choice, labels_val)
    chosen_raw_p = raw_probs[np.arange(len(bank_val)), argmax_choice]
    chosen_cal_p = calibrated_probs[np.arange(len(bank_val)), selected_choice]
    question_ece_argmax, question_bins_argmax = expected_calibration_error(chosen_raw_p[hard_val], argmax_hits[hard_val].astype(float))
    question_ece_selected, question_bins_selected = expected_calibration_error(chosen_cal_p[hard_val], selected_hits[hard_val].astype(float))

    config = {
        "base_weights": "dev/runs/exp_b08/arm_embeddings_plus_g01_g03.npz",
        "probability": "sigmoid(logit / T)",
        "temperature": temperature,
        "gamma": selected["gamma"],
        "gamma_grid": list(GAMMA_GRID),
        "selection_rule": "max val_hard correct among calibrated gammas whose val_all correct is at least the argmax count; ties use higher val_all, then smaller gamma",
        "g01_columns": g01_columns,
        "g03_columns": g03_columns,
    }
    config_path = runs / "exp_b09" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_bytes = config_path.stat().st_size
    base_bytes = uncompressed_weight_bytes(weights)

    audit = {
        "experiment": "EXP-B09",
        "research_question": "RQ4",
        "base_checkpoint": "dev/runs/exp_b08/arm_embeddings_plus_g01_g03.npz",
        "retrained": False,
        "probability": "sigmoid(logit / T)",
        "temperature": temperature,
        "gamma_grid": list(GAMMA_GRID),
        "nll": {"T1": nll_before, "calibrated": nll_after},
        "candidate_ece": {
            "val_all_T1": ece_before,
            "val_all_calibrated": ece_after,
            "val_hard_candidates_T1": ece_hard_before,
            "val_hard_candidates_calibrated": ece_hard_after,
            "bins_val_all_T1": bins_before,
            "bins_val_all_calibrated": bins_after,
        },
        "argmax": argmax_record,
        "sweep_T1": sweeps["temperature_1"],
        "sweep_calibrated": sweeps["calibrated"],
        "selected": selected,
        "official_validation_score": official,
        "question_level_val_hard": {
            "argmax_T1_ece": question_ece_argmax,
            "argmax_T1_bins": question_bins_argmax,
            "argmax_T1_quartiles": quartile_reliability(chosen_raw_p[hard_val], argmax_hits[hard_val]),
            "selected_ece": question_ece_selected,
            "selected_bins": question_bins_selected,
            "selected_quartiles": quartile_reliability(chosen_cal_p[hard_val], selected_hits[hard_val]),
            "mean_confidence_when_argmax_correct": float(chosen_raw_p[hard_val][argmax_hits[hard_val]].mean()),
            "mean_confidence_when_argmax_wrong": float(chosen_raw_p[hard_val][~argmax_hits[hard_val]].mean()),
        },
        "disagreement_val_hard": {
            "majority_vs_argmax": disagreement(b01_choice, argmax_choice, labels_val, hard_val),
            "argmax_vs_selected": disagreement(argmax_choice, selected_choice, labels_val, hard_val),
            "majority_vs_selected": disagreement(b01_choice, selected_choice, labels_val, hard_val),
        },
        "resources": {
            "forward_seconds_val_all_numpy": forward_seconds,
            "sweep_seconds_val_all": sweep_seconds,
            "base_uncompressed_bytes": base_bytes,
            "config_bytes": config_bytes,
            "shipped_uncompressed_bytes": base_bytes + config_bytes,
        },
        "files": {
            "config": "dev/runs/exp_b09/config.json",
            "predictions": "dev/runs/predictions/EXP-B09_calibrated_predictions.json",
            "score": "dev/runs/scores/EXP-B09_calibrated_score.json",
        },
    }
    audit_path = runs / "audit_logs" / "exp_b09_calibration_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected gamma={selected['gamma']} T={temperature:.6f} "
        f"val_hard {selected['val_hard']['correct']}/{selected['val_hard']['total']} "
        f"val_all {selected['val_all']['correct']}/{selected['val_all']['total']}"
    )
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b09()
