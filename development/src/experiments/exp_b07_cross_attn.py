"""EXP-B07: Permutation-equivariant cross-candidate attention (RQ3).

Compares isolated scoring (EXP-B06) with listwise scoring of the same four
candidates. Two arms, both trained only on the training split:

  Arm A  embeddings_only: L2-normalized [1024]. Same information as B06.
  Arm B  embeddings_plus_g04: Arm A plus the FEAT-G04 columns listed in
         private_configuration_omitted.json. Column names are read from that file.

MOD-M03 warms up on train_all and finetunes on train_hard. Five-fold splits
stay inside the training questions and are not used to pick a validation
winner. Validation is reported after the hyperparameters are already fixed.
Test references are not opened.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
from features.geometry import GeometryFeatureExtractor  # noqa: E402
from models.cross_attn_net import (  # noqa: E402
    CrossCandidateAttentionModule,
    CrossCandidateAttentionNet,
    loss_components,
    state_dict_to_numpy,
    uncompressed_weight_bytes,
)

PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
HPARAMS = {
    "seed": 42,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.1,
    "lambda_bce": 1.0,
    "weight_decay": 1e-2,
    "lr_warmup": 1e-3,
    "lr_finetune": 3e-4,
    "warmup_epochs": 10,
    "finetune_epochs": 8,
    "batch_size": 256,
    "grad_clip": 1.0,
    "n_folds": 5,
    "layernorm_eps": 1e-5,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)


def standardize_fit(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def selection_hits(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    chosen = np.argmax(scores, axis=1)
    return labels[np.arange(labels.shape[0]), chosen].astype(bool)


def accuracy_of(hits: np.ndarray) -> Dict[str, Any]:
    total = int(hits.shape[0])
    correct = int(hits.sum())
    return {"acc": (correct / total) if total else 0.0, "correct": correct, "total": total}


def kfold_indices(n_rows: int, n_folds: int, seed: int) -> List[np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return [np.sort(fold) for fold in np.array_split(permutation, n_folds)]


def _question_weight_tensor(features: np.ndarray, sample_weight: np.ndarray) -> torch.Tensor:
    weight = np.asarray(sample_weight, dtype=np.float32)
    if weight.ndim != 1 or weight.shape[0] != features.shape[0]:
        raise ValueError(
            f"sample_weight shape {weight.shape} does not match {features.shape[0]} questions"
        )
    if not np.isfinite(weight).all() or float(weight.min()) <= 0.0:
        raise ValueError("sample weights must be finite and positive")
    return torch.from_numpy(np.ascontiguousarray(weight))


def _run_phase_weighted(
    model: CrossCandidateAttentionModule,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    lam: float,
    sample_weight: np.ndarray,
) -> Dict[str, float]:
    """Same batches and step count as run_phase. The loss is a weighted mean.

    Uniform weights reproduce the original batch mean. The weight sum is the
    denominator, so a mean weight other than 1 does not rescale the learning rate.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HPARAMS["weight_decay"])
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(labels.astype(np.float32))),
        _question_weight_tensor(features, sample_weight),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
        generator=generator,
    )
    last_loss = 0.0
    last_listwise = 0.0
    last_bce = 0.0
    steps = 0
    for _epoch in range(epochs):
        total_loss = 0.0
        total_listwise = 0.0
        total_bce = 0.0
        seen = 0.0
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_x)
            listwise, bce = loss_components(scores, batch_y)
            per_row = listwise + lam * bce
            weight_sum = batch_w.sum()
            loss = (batch_w * per_row).sum() / weight_sum
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            optimizer.step()
            steps += 1
            wsum = float(weight_sum.detach().item())
            total_loss += float(loss.detach().item()) * wsum
            total_listwise += float((batch_w * listwise).sum().detach().item())
            total_bce += float((batch_w * bce).sum().detach().item())
            seen += wsum
        last_loss = total_loss / seen
        last_listwise = total_listwise / seen
        last_bce = total_bce / seen
    return {
        "loss": last_loss,
        "listwise": last_listwise,
        "bce": last_bce,
        "optimizer_steps": steps,
        "weight_sum": float(np.asarray(sample_weight, dtype=np.float64).sum()),
        "row_count": int(features.shape[0]),
    }


def run_phase(
    model: CrossCandidateAttentionModule,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    lam: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if sample_weight is not None:
        return _run_phase_weighted(
            model, features, labels, epochs, lr, seed, device, lam, sample_weight,
        )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HPARAMS["weight_decay"])
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)),
        torch.from_numpy(np.ascontiguousarray(labels.astype(np.float32))),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
        generator=generator,
    )
    last_loss = 0.0
    last_listwise = 0.0
    last_bce = 0.0
    steps = 0
    for _epoch in range(epochs):
        total_loss = 0.0
        total_listwise = 0.0
        total_bce = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_x)
            listwise, bce = loss_components(scores, batch_y)
            loss = (listwise + lam * bce).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            optimizer.step()
            steps += 1
            rows = int(batch_x.shape[0])
            total_loss += float(loss.item()) * rows
            total_listwise += float(listwise.mean().item()) * rows
            total_bce += float(bce.mean().item()) * rows
            seen += rows
        last_loss = total_loss / seen
        last_listwise = total_listwise / seen
        last_bce = total_bce / seen
    return {"loss": last_loss, "listwise": last_listwise, "bce": last_bce, "optimizer_steps": steps}


def fit_attention(
    x_warm: np.ndarray,
    y_warm: np.ndarray,
    x_fine: np.ndarray,
    y_fine: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    set_seed(seed)
    model = CrossCandidateAttentionModule(
        input_dim=int(x_warm.shape[-1]),
        hidden_dim=HPARAMS["hidden_dim"],
        num_layers=HPARAMS["num_layers"],
        num_heads=HPARAMS["num_heads"],
        dropout=HPARAMS["dropout"],
        eps=HPARAMS["layernorm_eps"],
    ).to(device)
    warm_stats = run_phase(
        model, x_warm, y_warm,
        epochs=HPARAMS["warmup_epochs"],
        lr=HPARAMS["lr_warmup"],
        seed=seed,
        device=device,
        lam=HPARAMS["lambda_bce"],
    )
    fine_stats = run_phase(
        model, x_fine, y_fine,
        epochs=HPARAMS["finetune_epochs"],
        lr=HPARAMS["lr_finetune"],
        seed=seed + 1000,
        device=device,
        lam=HPARAMS["lambda_bce"],
    )
    model.eval()
    weights = state_dict_to_numpy(model)
    return weights, {"warmup": warm_stats, "finetune": fine_stats}


def score_with_numpy(weights: Dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return CrossCandidateAttentionNet(weights).forward_numpy(features)


def numpy_torch_gap(weights: Dict[str, np.ndarray], features: np.ndarray) -> float:
    """Max absolute gap between NumPy inference and a CPU PyTorch eval forward."""
    module = CrossCandidateAttentionModule(
        input_dim=int(features.shape[-1]),
        hidden_dim=HPARAMS["hidden_dim"],
        num_layers=HPARAMS["num_layers"],
        num_heads=HPARAMS["num_heads"],
        dropout=HPARAMS["dropout"],
        eps=HPARAMS["layernorm_eps"],
    )
    model_keys = set(module.state_dict().keys())
    state = {
        key: torch.from_numpy(np.asarray(value))
        for key, value in weights.items()
        if key in model_keys
    }
    module.load_state_dict(state)
    module.eval()
    with torch.no_grad():
        torch_scores = module(torch.from_numpy(np.ascontiguousarray(features))).numpy()
    numpy_scores = score_with_numpy(weights, features)
    return float(np.max(np.abs(torch_scores - numpy_scores)))


def cross_validate(
    arm_name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    hard_mask: np.ndarray,
    geometry: Optional[np.ndarray],
    folds: Sequence[np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    fold_rows = []
    for fold_id, held in enumerate(folds):
        train_idx = np.concatenate([folds[other] for other in range(len(folds)) if other != fold_id])
        hard_train = train_idx[hard_mask[train_idx]]
        hard_held = held[hard_mask[held]]
        mean = std = None
        if geometry is None:
            features = embeddings
        else:
            mean, std = standardize_fit(geometry[train_idx])
            features = np.concatenate(
                [embeddings, standardize_apply(geometry, mean, std)],
                axis=-1,
            ).astype(np.float32)
        started = time.perf_counter()
        weights, _stats = fit_attention(
            features[train_idx], labels[train_idx],
            features[hard_train], labels[hard_train],
            seed=HPARAMS["seed"] + fold_id,
            device=device,
        )
        elapsed = time.perf_counter() - started
        held_scores = score_with_numpy(weights, features[held])
        held_hits = selection_hits(held_scores, labels[held])
        hard_hits = selection_hits(score_with_numpy(weights, features[hard_held]), labels[hard_held])
        row = {
            "fold": fold_id,
            "train_questions": int(train_idx.size),
            "held_out_questions": int(held.size),
            "held_out_hard_questions": int(hard_held.size),
            "held_out_all": accuracy_of(held_hits),
            "held_out_hard": accuracy_of(hard_hits),
            "fit_seconds": elapsed,
        }
        fold_rows.append(row)
        print(
            f"  [{arm_name}] fold {fold_id}: held-out hard "
            f"{row['held_out_hard']['correct']}/{row['held_out_hard']['total']} "
            f"= {row['held_out_hard']['acc'] * 100:.4f}%  ({elapsed:.1f}s)"
        )
    hard_accs = [row["held_out_hard"]["acc"] for row in fold_rows]
    return {
        "arm": arm_name,
        "folds": fold_rows,
        "mean_held_out_hard_acc": float(np.mean(hard_accs)),
        "std_held_out_hard_acc": float(np.std(hard_accs)),
    }


def load_g04_columns() -> List[str]:
    contract_path = WORKSPACE / "dev" / "runs" / "audit_logs" / "private_configuration_omitted.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    columns = list(contract["statistically_eligible_by_family"]["FEAT-G04"])
    if not columns:
        raise RuntimeError("FEAT-G04 eligible column list is empty")
    return columns


def extract_g04(bank, arrays, columns: Sequence[str]) -> np.ndarray:
    extractor = GeometryFeatureExtractor()
    full = extractor.transform(bank, arrays)
    names = extractor.FEATURE_NAMES
    missing = [column for column in columns if column not in names]
    if missing:
        raise RuntimeError(f"contract columns missing from FEAT-G04: {missing}")
    index = [names.index(column) for column in columns]
    return full[:, :, index].astype(np.float32)


def write_official_predictions(path: Path, bank, chosen: np.ndarray) -> None:
    payload = {
        "protocol": PROTOCOL,
        "split": "validation",
        "ids": [row["id"] for row in bank],
        "selected": [int(index) for index in chosen],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def official_score(predictions: Path, output: Path) -> Dict[str, Any]:
    scorer = WORKSPACE / "DASE7506-MP2" / "code" / "score_predictions.py"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    subprocess.run(
        [sys.executable, str(scorer), "--split", "validation",
         "--predictions", str(predictions), "--output", str(output)],
        check=True,
        env=env,
        cwd=str(WORKSPACE),
    )
    return json.loads(output.read_text(encoding="utf-8"))


def repair_report(reference_hits: np.ndarray, new_hits: np.ndarray, ids: Sequence[str], hard_mask: np.ndarray) -> Dict[str, Any]:
    hard_ids = [ids[i] for i in range(len(ids)) if hard_mask[i]]
    ref = reference_hits[hard_mask]
    new = new_hits[hard_mask]
    rescued_at = [hard_ids[i] for i in range(len(hard_ids)) if (not ref[i]) and new[i]]
    regressed_at = [hard_ids[i] for i in range(len(hard_ids)) if ref[i] and (not new[i])]
    return {
        "b06_hard_correct": int(ref.sum()),
        "b06_hard_wrong": int((~ref).sum()),
        "rescued": int((~ref & new).sum()),
        "regressed": int((ref & ~new).sum()),
        "both_correct": int((ref & new).sum()),
        "both_wrong": int((~ref & ~new).sum()),
        "rescued_example_ids": rescued_at[:5],
        "regressed_example_ids": regressed_at[:5],
    }


def evaluate_arm(
    arm_name: str,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    hard_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    hard_val: np.ndarray,
    seed: int,
    device: torch.device,
    extra_weights: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    weights, train_stats = fit_attention(
        features_train, labels_train,
        features_train[hard_train], labels_train[hard_train],
        seed=seed,
        device=device,
    )
    fit_seconds = time.perf_counter() - started
    if extra_weights:
        weights.update(extra_weights)
    parity = numpy_torch_gap(weights, features_val[:32])
    if parity > 1e-4:
        raise RuntimeError(f"{arm_name} NumPy/Torch gap {parity} exceeds 1e-4")

    train_hard_scores = score_with_numpy(weights, features_train[hard_train])
    val_scores = score_with_numpy(weights, features_val)
    val_hits = selection_hits(val_scores, labels_val)
    hard_hits = val_hits[hard_val]
    train_hits = selection_hits(train_hard_scores, labels_train[hard_train])

    timer_input = np.ascontiguousarray(features_val)
    net = CrossCandidateAttentionNet(weights)
    net.forward_numpy(timer_input[:8])
    tick = time.perf_counter()
    timed_scores = net.forward_numpy(timer_input)
    forward_seconds = time.perf_counter() - tick
    if not np.array_equal(np.argmax(timed_scores, axis=1), np.argmax(val_scores, axis=1)):
        raise RuntimeError(f"{arm_name} timed forward disagreed with recorded scores")

    return {
        "arm": arm_name,
        "input_dim": int(features_train.shape[-1]),
        "weights": weights,
        "train_stats": train_stats,
        "fit_seconds": fit_seconds,
        "numpy_torch_max_abs": parity,
        "forward_seconds_val_all": forward_seconds,
        "uncompressed_bytes": uncompressed_weight_bytes(weights),
        "accuracy": {
            "val_hard": accuracy_of(hard_hits),
            "val_all": accuracy_of(val_hits),
            "train_hard": accuracy_of(train_hits),
        },
        "val_hits": val_hits,
        "val_chosen": np.argmax(val_scores, axis=1).astype(int),
        "val_choice_histogram": np.bincount(np.argmax(val_scores, axis=1), minlength=4).tolist(),
    }


def save_weights(path: Path, weights: Dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def run_exp_b07() -> Dict[str, Any]:
    print("=" * 70)
    print("Executing EXP-B07: Cross-candidate attention (RQ3)")
    print("=" * 70)
    torch.set_num_threads(4)
    device = pick_device()
    print(f"Train device: {device}. Frozen scores use CPU NumPy.")

    bank_train, arrays_train = load_dataset("train")
    gold_train = load_gold_references("train")
    labels_train = build_label_matrix(bank_train, gold_train)
    hard_train = get_hard_subset_mask(bank_train)

    bank_val, arrays_val = load_dataset("validation")
    gold_val = load_gold_references("validation")
    labels_val = build_label_matrix(bank_val, gold_val)
    hard_val = get_hard_subset_mask(bank_val)

    if labels_train.sum(axis=1).min() < 1 or labels_val.sum(axis=1).min() < 1:
        raise RuntimeError("a question has no correct candidate")
    print(
        f"train_all={len(bank_train)} train_hard={int(hard_train.sum())} "
        f"val_all={len(bank_val)} val_hard={int(hard_val.sum())}"
    )

    emb_train = l2_normalize(arrays_train["embeddings"])
    emb_val = l2_normalize(arrays_val["embeddings"])
    g04_columns = load_g04_columns()
    print(f"FEAT-G04 columns from contract: {g04_columns}")
    print("Extracting FEAT-G04...")
    g04_train = extract_g04(bank_train, arrays_train, g04_columns)
    g04_val = extract_g04(bank_val, arrays_val, g04_columns)

    folds = kfold_indices(len(bank_train), HPARAMS["n_folds"], HPARAMS["seed"])
    print("\n--- 5-fold inside training questions (not used to pick a val winner) ---")
    cv_a = cross_validate("embeddings_only", emb_train, labels_train, hard_train, None, folds, device)
    cv_b = cross_validate("embeddings_plus_g04", emb_train, labels_train, hard_train, g04_train, folds, device)

    print("\n--- Final fit on train_all warmup + train_hard finetune ---")
    arm_a = evaluate_arm(
        "embeddings_only", emb_train, labels_train, hard_train,
        emb_val, labels_val, hard_val, HPARAMS["seed"], device,
    )
    mean_b, std_b = standardize_fit(g04_train)
    features_train_b = np.concatenate([emb_train, standardize_apply(g04_train, mean_b, std_b)], axis=-1)
    features_val_b = np.concatenate([emb_val, standardize_apply(g04_val, mean_b, std_b)], axis=-1)
    arm_b = evaluate_arm(
        "embeddings_plus_g04", features_train_b, labels_train, hard_train,
        features_val_b, labels_val, hard_val, HPARAMS["seed"], device,
        extra_weights={
            "g04_mean": mean_b,
            "g04_std": std_b,
            "g04_column_count": np.asarray([len(g04_columns)], dtype=np.int64),
        },
    )

    runs = WORKSPACE / "dev" / "runs"
    weight_dir = runs / "exp_b07"
    pred_dir = runs / "predictions"
    score_dir = runs / "scores"
    audit_dir = runs / "audit_logs"
    for directory in (weight_dir, pred_dir, score_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    path_a = weight_dir / "arm_embeddings_only.npz"
    path_b = weight_dir / "arm_embeddings_plus_g04.npz"
    hash_a = save_weights(path_a, arm_a["weights"])
    hash_b = save_weights(path_b, arm_b["weights"])

    pred_a = pred_dir / "EXP-B07_cross_attn_predictions.json"
    pred_b = pred_dir / "EXP-B07_g04_predictions.json"
    write_official_predictions(pred_a, bank_val, arm_a["val_chosen"])
    write_official_predictions(pred_b, bank_val, arm_b["val_chosen"])
    official_a = official_score(pred_a, score_dir / "EXP-B07_cross_attn_score.json")
    official_b = official_score(pred_b, score_dir / "EXP-B07_g04_score.json")

    if int(official_a["correct"]) != arm_a["accuracy"]["val_all"]["correct"]:
        raise RuntimeError(
            f"official scorer {official_a['correct']} != internal {arm_a['accuracy']['val_all']['correct']}"
        )
    if int(official_b["correct"]) != arm_b["accuracy"]["val_all"]["correct"]:
        raise RuntimeError(
            f"official scorer {official_b['correct']} != internal {arm_b['accuracy']['val_all']['correct']}"
        )

    b06_path = pred_dir / "EXP-B06_pointwise_predictions.json"
    b06_chosen = np.asarray(json.loads(b06_path.read_text(encoding="utf-8"))["predictions"], dtype=int)
    if b06_chosen.shape[0] != len(bank_val):
        raise RuntimeError("EXP-B06 predictions are not aligned with the validation bank")
    b06_hits = selection_hits(
        np.eye(4, dtype=np.float32)[b06_chosen],
        labels_val,
    )
    # selection_hits argmaxes a one-hot score matrix, which recovers b06_chosen.
    if not np.array_equal(np.argmax(np.eye(4, dtype=np.float32)[b06_chosen], axis=1), b06_chosen):
        raise RuntimeError("B06 index recovery failed")
    val_ids = [row["id"] for row in bank_val]
    repair_a = repair_report(b06_hits, arm_a["val_hits"], val_ids, hard_val)
    repair_b = repair_report(b06_hits, arm_b["val_hits"], val_ids, hard_val)

    def public_arm(result: Dict[str, Any], weight_path: Path, digest: str, official: Dict[str, Any], repair: Dict[str, Any]) -> Dict[str, Any]:
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
            "repair_vs_b06_val_hard": repair,
        }

    audit = {
        "experiment": "EXP-B07",
        "research_question": "RQ3",
        "protocol": PROTOCOL,
        "device_train": str(device),
        "inference": "cpu numpy",
        "hyperparameters": HPARAMS,
        "g04_columns": g04_columns,
        "g04_standardizer_fit_on": "train_all",
        "lambda_note": "lambda_bce=1 follows the written loss. It was not selected on validation.",
        "cv_note": "5-fold held-out hard accuracy is a training-split stability check. The final models are refit on all training questions.",
        "counts": {
            "train_all": len(bank_train),
            "train_hard": int(hard_train.sum()),
            "val_all": len(bank_val),
            "val_hard": int(hard_val.sum()),
        },
        "cross_validation": {"embeddings_only": cv_a, "embeddings_plus_g04": cv_b},
        "arms": {
            "embeddings_only": public_arm(arm_a, path_a, hash_a, official_a, repair_a),
            "embeddings_plus_g04": public_arm(arm_b, path_b, hash_b, official_b, repair_b),
        },
        "baselines_val_hard": {
            "majority_voting_b01": 0.690141,
            "structured_logreg_b03": 0.732394,
            "pointwise_b06": 0.785915,
        },
        "files": {
            "weights_embeddings_only": str(path_a.relative_to(WORKSPACE)),
            "weights_embeddings_plus_g04": str(path_b.relative_to(WORKSPACE)),
            "predictions_embeddings_only": str(pred_a.relative_to(WORKSPACE)),
            "predictions_embeddings_plus_g04": str(pred_b.relative_to(WORKSPACE)),
            "score_embeddings_only": "dev/runs/scores/EXP-B07_cross_attn_score.json",
            "score_embeddings_plus_g04": "dev/runs/scores/EXP-B07_g04_score.json",
        },
    }
    audit_path = audit_dir / "exp_b07_cross_attn_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, result in (("embeddings_only", arm_a), ("embeddings_plus_g04", arm_b)):
        acc = result["accuracy"]
        print(
            f"{name}: val_hard {acc['val_hard']['correct']}/{acc['val_hard']['total']} "
            f"= {acc['val_hard']['acc'] * 100:.4f}% | val_all {acc['val_all']['correct']}/{acc['val_all']['total']} "
            f"= {acc['val_all']['acc'] * 100:.4f}% | bytes {result['uncompressed_bytes']}"
        )
    print(f"Audit saved to: {audit_path}")
    return audit


if __name__ == "__main__":
    run_exp_b07()
