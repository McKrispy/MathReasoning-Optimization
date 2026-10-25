"""EXP-B08: Does explicit reasoning verification help? (RQ1)

Same MOD-M03 training recipe as EXP-B07. The 1024-d embeddings stay
L2-normalized. Structured columns are read from the B02 contract:

  Arm A  embeddings_plus_g01:
         embeddings plus the FEAT-G01 consensus core (eligible G01 columns
         after removing gsm8k_morphology_columns_not_core).
  Arm B  embeddings_plus_g01_g03:
         Arm A plus every statistically eligible FEAT-G03 column.
         This is the batch's specified model.

Morphology columns, FEAT-G02 and FEAT-G04 stay out. Hyperparameters are the
B07 constants. Five-fold splits stay inside the training questions. Test
references are not opened.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import (  # noqa: E402
    build_label_matrix,
    get_hard_subset_mask,
    load_dataset,
    load_gold_references,
)
from features.consensus import ConsensusFeatureExtractor  # noqa: E402
from features.symbolic import SymbolicVerificationFeatureExtractor  # noqa: E402
from experiments.exp_b07_cross_attn import (  # noqa: E402
    HPARAMS,
    accuracy_of,
    cross_validate,
    evaluate_arm,
    kfold_indices,
    l2_normalize,
    official_score,
    pick_device,
    standardize_apply,
    standardize_fit,
    write_official_predictions,
)


def select_reasoning_columns(contract: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Consensus core, then the eligible symbolic columns. Names come from the contract."""
    morphology = set(contract["gsm8k_morphology_columns_not_core"])
    g01 = [
        column for column in contract["statistically_eligible_by_family"]["FEAT-G01"]
        if column not in morphology
    ]
    g03 = list(contract["statistically_eligible_by_family"]["FEAT-G03"])
    if not g01 or not g03:
        raise RuntimeError("contract returned an empty G01 core or G03 list")
    leaked = [column for column in g01 if column in morphology]
    if leaked:
        raise RuntimeError(f"morphology columns entered the G01 core: {leaked}")
    return g01, g03


def take_columns(features: np.ndarray, names: Sequence[str], columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in names]
    if missing:
        raise RuntimeError(f"columns missing from extractor: {missing}")
    index = [list(names).index(column) for column in columns]
    return features[:, :, index].astype(np.float32)


def load_contract() -> Dict[str, Any]:
    path = WORKSPACE / "dev" / "runs" / "audit_logs" / "private_configuration_omitted.json"
    return json.loads(path.read_text(encoding="utf-8"))


def hits_from_official(path: Path, bank, labels: np.ndarray) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    choice_by_id = dict(zip(payload["ids"], payload["selected"]))
    chosen = np.asarray([int(choice_by_id[row["id"]]) for row in bank], dtype=int)
    return labels[np.arange(labels.shape[0]), chosen].astype(bool)


def compare_hits(reference: np.ndarray, new: np.ndarray, ids: Sequence[str], mask: np.ndarray) -> Dict[str, Any]:
    masked_ids = [ids[i] for i in range(len(ids)) if mask[i]]
    ref = reference[mask]
    cur = new[mask]
    rescued = [masked_ids[i] for i in range(len(masked_ids)) if (not ref[i]) and cur[i]]
    regressed = [masked_ids[i] for i in range(len(masked_ids)) if ref[i] and (not cur[i])]
    return {
        "reference_correct": int(ref.sum()),
        "reference_wrong": int((~ref).sum()),
        "rescued": int((~ref & cur).sum()),
        "regressed": int((ref & ~cur).sum()),
        "both_correct": int((ref & cur).sum()),
        "both_wrong": int((~ref & ~cur).sum()),
        "rescued_example_ids": rescued[:5],
        "regressed_example_ids": regressed[:5],
    }


def row_disagrees(column: np.ndarray) -> np.ndarray:
    return column.max(axis=1) > column.min(axis=1)


def save_weights(path: Path, weights: Dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_arm(result: Dict[str, Any], weight_path: Path, digest: str, official: Dict[str, Any], versus: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm": result["arm"],
        "input_dim": result["input_dim"],
        "accuracy": result["accuracy"],
        "choice_histogram_val_all": result["val_choice_histogram"],
        "fit_seconds": result["fit_seconds"],
        "train_stats": result["train_stats"],
        "forward_seconds_val_all_numpy": result["forward_seconds_val_all"],
        "uncompressed_bytes": result["uncompressed_bytes"],
        "uncompressed_mib": result["uncompressed_bytes"] / (1024 ** 2),
        "weight_file_bytes": weight_path.stat().st_size,
        "sha256": digest,
        "numpy_torch_max_abs": result["numpy_torch_max_abs"],
        "official_validation_score": official,
        "comparisons_val_hard": versus,
    }


def run_exp_b08() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B08: Reasoning verification on MOD-M03 (RQ1)")
    print("=" * 70)
    torch.set_num_threads(4)
    device = pick_device()
    print(f"Train device: {device}. Frozen scores use CPU NumPy.")

    contract = load_contract()
    g01_columns, g03_columns = select_reasoning_columns(contract)
    print(f"G01 core: {g01_columns}")
    print(f"G03 eligible: {g03_columns}")

    bank_train, arrays_train = load_dataset("train")
    labels_train = build_label_matrix(bank_train, load_gold_references("train"))
    hard_train = get_hard_subset_mask(bank_train)
    bank_val, arrays_val = load_dataset("validation")
    labels_val = build_label_matrix(bank_val, load_gold_references("validation"))
    hard_val = get_hard_subset_mask(bank_val)
    if labels_train.sum(axis=1).min() < 1 or labels_val.sum(axis=1).min() < 1:
        raise RuntimeError("a question has no correct candidate")
    print(
        f"train_all={len(bank_train)} train_hard={int(hard_train.sum())} "
        f"val_all={len(bank_val)} val_hard={int(hard_val.sum())}"
    )

    emb_train = l2_normalize(arrays_train["embeddings"])
    emb_val = l2_normalize(arrays_val["embeddings"])
    print("Extracting FEAT-G01 and FEAT-G03...")
    tick = time.perf_counter()
    g01_train_full = ConsensusFeatureExtractor().transform(bank_train, arrays_train)
    g03_train_full = SymbolicVerificationFeatureExtractor().transform(bank_train, arrays_train)
    train_feature_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    g01_val_full = ConsensusFeatureExtractor().transform(bank_val, arrays_val)
    g03_val_full = SymbolicVerificationFeatureExtractor().transform(bank_val, arrays_val)
    val_feature_seconds = time.perf_counter() - tick
    print(f"Feature extraction: train {train_feature_seconds:.3f}s, validation {val_feature_seconds:.3f}s")

    g01_names = ConsensusFeatureExtractor.FEATURE_NAMES
    g03_names = SymbolicVerificationFeatureExtractor.FEATURE_NAMES
    g01_train = take_columns(g01_train_full, g01_names, g01_columns)
    g01_val = take_columns(g01_val_full, g01_names, g01_columns)
    g03_train = take_columns(g03_train_full, g03_names, g03_columns)
    g03_val = take_columns(g03_val_full, g03_names, g03_columns)
    struct_train = np.concatenate([g01_train, g03_train], axis=-1).astype(np.float32)
    struct_val = np.concatenate([g01_val, g03_val], axis=-1).astype(np.float32)

    folds = kfold_indices(len(bank_train), HPARAMS["n_folds"], HPARAMS["seed"])
    print("\n--- 5-fold inside training questions ---")
    cv_a = cross_validate("embeddings_plus_g01", emb_train, labels_train, hard_train, g01_train, folds, device)
    cv_b = cross_validate("embeddings_plus_g01_g03", emb_train, labels_train, hard_train, struct_train, folds, device)

    print("\n--- Final fit on train_all warmup + train_hard finetune ---")
    mean_a, std_a = standardize_fit(g01_train)
    features_train_a = np.concatenate([emb_train, standardize_apply(g01_train, mean_a, std_a)], axis=-1)
    features_val_a = np.concatenate([emb_val, standardize_apply(g01_val, mean_a, std_a)], axis=-1)
    arm_a = evaluate_arm(
        "embeddings_plus_g01", features_train_a, labels_train, hard_train,
        features_val_a, labels_val, hard_val, HPARAMS["seed"], device,
        extra_weights={
            "struct_mean": mean_a,
            "struct_std": std_a,
            "struct_column_count": np.asarray([len(g01_columns)], dtype=np.int64),
        },
    )
    mean_b, std_b = standardize_fit(struct_train)
    features_train_b = np.concatenate([emb_train, standardize_apply(struct_train, mean_b, std_b)], axis=-1)
    features_val_b = np.concatenate([emb_val, standardize_apply(struct_val, mean_b, std_b)], axis=-1)
    arm_b = evaluate_arm(
        "embeddings_plus_g01_g03", features_train_b, labels_train, hard_train,
        features_val_b, labels_val, hard_val, HPARAMS["seed"], device,
        extra_weights={
            "struct_mean": mean_b,
            "struct_std": std_b,
            "struct_column_count": np.asarray([struct_train.shape[-1]], dtype=np.int64),
        },
    )

    runs = WORKSPACE / "dev" / "runs"
    weight_dir = runs / "exp_b08"
    pred_dir = runs / "predictions"
    score_dir = runs / "scores"
    audit_dir = runs / "audit_logs"
    for directory in (weight_dir, pred_dir, score_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    path_a = weight_dir / "arm_embeddings_plus_g01.npz"
    path_b = weight_dir / "arm_embeddings_plus_g01_g03.npz"
    hash_a = save_weights(path_a, arm_a["weights"])
    hash_b = save_weights(path_b, arm_b["weights"])
    pred_a = pred_dir / "EXP-B08_g01_predictions.json"
    pred_b = pred_dir / "EXP-B08_reasoning_predictions.json"
    write_official_predictions(pred_a, bank_val, arm_a["val_chosen"])
    write_official_predictions(pred_b, bank_val, arm_b["val_chosen"])
    official_a = official_score(pred_a, score_dir / "EXP-B08_g01_score.json")
    official_b = official_score(pred_b, score_dir / "EXP-B08_reasoning_score.json")
    if int(official_a["correct"]) != arm_a["accuracy"]["val_all"]["correct"]:
        raise RuntimeError("arm A official score disagrees with internal val_all")
    if int(official_b["correct"]) != arm_b["accuracy"]["val_all"]["correct"]:
        raise RuntimeError("arm B official score disagrees with internal val_all")

    val_ids = [row["id"] for row in bank_val]
    b01_hits = hits_from_official(pred_dir / "EXP-B01_validation_predictions.json", bank_val, labels_val)
    b07_hits = hits_from_official(pred_dir / "EXP-B07_cross_attn_predictions.json", bank_val, labels_val)
    versus_a = {
        "versus_b01_majority": compare_hits(b01_hits, arm_a["val_hits"], val_ids, hard_val),
        "versus_b07_embeddings": compare_hits(b07_hits, arm_a["val_hits"], val_ids, hard_val),
    }
    versus_b = {
        "versus_b01_majority": compare_hits(b01_hits, arm_b["val_hits"], val_ids, hard_val),
        "versus_b07_embeddings": compare_hits(b07_hits, arm_b["val_hits"], val_ids, hard_val),
        "versus_g01_arm": compare_hits(arm_a["val_hits"], arm_b["val_hits"], val_ids, hard_val),
    }

    arith_index = g03_columns.index("has_arithmetic_error")
    missing_index = g03_columns.index("missing_condition_count")
    arith_varies = row_disagrees(g03_val[:, :, arith_index])
    missing_varies = row_disagrees(g03_val[:, :, missing_index])
    reasoning_varies = arith_varies | missing_varies

    def subset_report(mask: np.ndarray) -> Dict[str, Any]:
        focused = mask & hard_val
        return {
            "questions": int(focused.sum()),
            "b01_majority": accuracy_of(b01_hits[focused]),
            "b07_embeddings": accuracy_of(b07_hits[focused]),
            "embeddings_plus_g01": accuracy_of(arm_a["val_hits"][focused]),
            "embeddings_plus_g01_g03": accuracy_of(arm_b["val_hits"][focused]),
        }

    audit = {
        "experiment": "EXP-B08",
        "research_question": "RQ1",
        "device_train": str(device),
        "inference": "cpu numpy",
        "hyperparameters": HPARAMS,
        "g01_columns": g01_columns,
        "g03_columns": g03_columns,
        "excluded": [
            "gsm8k_morphology_columns_not_core",
            "FEAT-G02",
            "FEAT-G04",
        ],
        "standardizer_fit_on": "train_all",
        "lambda_note": "lambda_bce=1 is the B07 constant. It was not selected on validation.",
        "counts": {
            "train_all": len(bank_train),
            "train_hard": int(hard_train.sum()),
            "val_all": len(bank_val),
            "val_hard": int(hard_val.sum()),
        },
        "feature_seconds": {
            "train_g01_g03": train_feature_seconds,
            "validation_g01_g03": val_feature_seconds,
        },
        "cross_validation": {
            "embeddings_plus_g01": cv_a,
            "embeddings_plus_g01_g03": cv_b,
        },
        "arms": {
            "embeddings_plus_g01": public_arm(arm_a, path_a, hash_a, official_a, versus_a),
            "embeddings_plus_g01_g03": public_arm(arm_b, path_b, hash_b, official_b, versus_b),
        },
        "reasoning_disagreement_slices": {
            "val_hard_arithmetic_error_disagreement": subset_report(arith_varies),
            "val_hard_missing_condition_disagreement": subset_report(missing_varies),
            "val_hard_either_reasoning_disagreement": subset_report(reasoning_varies),
            "val_hard_no_reasoning_disagreement": subset_report(~reasoning_varies),
        },
        "references_val_hard": {
            "majority_voting_b01": accuracy_of(b01_hits[hard_val]),
            "embeddings_attention_b07": accuracy_of(b07_hits[hard_val]),
        },
        "files": {
            "weights_g01": str(path_a.relative_to(WORKSPACE)),
            "weights_g01_g03": str(path_b.relative_to(WORKSPACE)),
            "predictions_g01": str(pred_a.relative_to(WORKSPACE)),
            "predictions_g01_g03": str(pred_b.relative_to(WORKSPACE)),
            "score_g01": "dev/runs/scores/EXP-B08_g01_score.json",
            "score_g01_g03": "dev/runs/scores/EXP-B08_reasoning_score.json",
        },
    }
    audit_path = audit_dir / "exp_b08_reasoning_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, result in (("embeddings_plus_g01", arm_a), ("embeddings_plus_g01_g03", arm_b)):
        acc = result["accuracy"]
        print(
            f"{name}: val_hard {acc['val_hard']['correct']}/{acc['val_hard']['total']} "
            f"= {acc['val_hard']['acc'] * 100:.4f}% | val_all {acc['val_all']['correct']}/{acc['val_all']['total']} "
            f"= {acc['val_all']['acc'] * 100:.4f}%"
        )
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b08()
