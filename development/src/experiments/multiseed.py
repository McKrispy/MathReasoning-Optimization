"""Shared multi-seed machinery for EXP-B12 ~ EXP-B14.

Seeds, derived seed streams, deterministic CUDA settings, fold models,
POST-P01/P02 decoding and the paired retention rule all follow the
"EXP-B12 ~ EXP-B15 执行约定" block in private_document_omitted.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

DEV_SRC_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
if str(DEV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_SRC_DIR))

from common_utils import build_label_matrix, get_hard_subset_mask, load_dataset, load_gold_references  # noqa: E402
from features.consensus import ConsensusFeatureExtractor  # noqa: E402
from features.symbolic import SymbolicVerificationFeatureExtractor  # noqa: E402
from models.cross_attn_net import (  # noqa: E402
    CrossCandidateAttentionModule,
    CrossCandidateAttentionNet,
    state_dict_to_numpy,
    uncompressed_weight_bytes,
)
from postprocessing.aggregation import WeightedSelfConsistencyAggregator  # noqa: E402
from postprocessing.calibration import TemperatureScaler  # noqa: E402
from experiments.exp_b07_cross_attn import (  # noqa: E402
    HPARAMS,
    accuracy_of,
    kfold_indices,
    l2_normalize,
    run_phase,
    standardize_apply,
    standardize_fit,
)
from experiments.exp_b08_reasoning import load_contract, select_reasoning_columns, take_columns  # noqa: E402
from experiments.exp_b09_calibration import GAMMA_GRID, select_gamma  # noqa: E402

SEEDS: Tuple[int, ...] = (42, 43, 44)
FOLD_PARTITION_SEED = 42
RETAIN_MIN_MEAN_GAIN = 3.55
RETAIN_MAX_VAL_ALL_DROP = 4.0
RUNS = WORKSPACE / "dev" / "runs"
REGISTRY_PATH = RUNS / "audit_logs" / "seed_registry.json"


def full_model_seeds(seed: int) -> Dict[str, int]:
    return {"warmup": seed, "finetune": seed + 1000}


def fold_model_seeds(seed: int, fold: int) -> Dict[str, int]:
    return {"warmup": 100 * seed + fold, "finetune": 100 * seed + fold + 1000}


def configure_determinism() -> torch.device:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_record(device: torch.device) -> Dict[str, Any]:
    import sklearn
    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device_train": str(device),
        "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if device.type == "cuda":
        record["gpu"] = torch.cuda.get_device_name(0)
    try:
        import lightgbm
        record["lightgbm"] = lightgbm.__version__
    except ImportError:
        record["lightgbm"] = None
    return record


def train_model(
    x_warm: np.ndarray,
    y_warm: np.ndarray,
    x_fine: np.ndarray,
    y_fine: np.ndarray,
    seeds: Dict[str, int],
    device: torch.device,
    module_factory: Optional[Callable[[int], torch.nn.Module]] = None,
    extra_warm: Optional[np.ndarray] = None,
    extra_fine: Optional[np.ndarray] = None,
    sample_weight_warm: Optional[np.ndarray] = None,
    sample_weight_fine: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """MOD-M03 recipe from EXP-B07 with explicit seed streams.

    extra_* carry a per-question side input (MOD-M05 same-answer matrix).
    sample_weight_* are per-question loss weights. Omit both to keep the
    original unweighted mean. They are not combined with side inputs.
    """
    if (sample_weight_warm is None) != (sample_weight_fine is None):
        raise ValueError("sample weights must be supplied for both phases or for neither")
    if sample_weight_warm is not None and extra_warm is not None:
        raise ValueError("sample weights are not combined with side inputs")
    np.random.seed(seeds["warmup"])
    torch.manual_seed(seeds["warmup"])
    input_dim = int(x_warm.shape[-1])
    if module_factory is None:
        model = CrossCandidateAttentionModule(
            input_dim=input_dim,
            hidden_dim=HPARAMS["hidden_dim"],
            num_layers=HPARAMS["num_layers"],
            num_heads=HPARAMS["num_heads"],
            dropout=HPARAMS["dropout"],
            eps=HPARAMS["layernorm_eps"],
        )
    else:
        model = module_factory(input_dim)
    model = model.to(device)
    started = time.perf_counter()
    if extra_warm is None:
        warm = run_phase(model, x_warm, y_warm, HPARAMS["warmup_epochs"], HPARAMS["lr_warmup"],
                         seeds["warmup"], device, HPARAMS["lambda_bce"],
                         sample_weight=sample_weight_warm)
        fine = run_phase(model, x_fine, y_fine, HPARAMS["finetune_epochs"], HPARAMS["lr_finetune"],
                         seeds["finetune"], device, HPARAMS["lambda_bce"],
                         sample_weight=sample_weight_fine)
    else:
        from experiments.exp_b14_answer_aware import run_phase_with_side
        warm = run_phase_with_side(model, x_warm, extra_warm, y_warm, HPARAMS["warmup_epochs"],
                                   HPARAMS["lr_warmup"], seeds["warmup"], device)
        fine = run_phase_with_side(model, x_fine, extra_fine, y_fine, HPARAMS["finetune_epochs"],
                                   HPARAMS["lr_finetune"], seeds["finetune"], device)
    model.eval()
    return state_dict_to_numpy(model), {
        "warmup": warm,
        "finetune": fine,
        "fit_seconds": time.perf_counter() - started,
    }


def default_scorer(weights: Dict[str, np.ndarray], features: np.ndarray, side: Optional[np.ndarray] = None) -> np.ndarray:
    return CrossCandidateAttentionNet(weights).forward_numpy(features)


# ---------------------------------------------------------------- data


class SplitData:
    def __init__(self, split: str):
        self.bank, self.arrays = load_dataset(split)
        self.labels = build_label_matrix(self.bank, load_gold_references(split))
        self.hard = get_hard_subset_mask(self.bank)
        self.emb = l2_normalize(self.arrays["embeddings"])
        if self.labels.sum(axis=1).min() < 1:
            raise RuntimeError(f"{split}: a question has no correct candidate")


def reasoning_struct(split: SplitData) -> Tuple[np.ndarray, List[str], List[str]]:
    g01_columns, g03_columns = select_reasoning_columns(load_contract())
    g01 = take_columns(ConsensusFeatureExtractor().transform(split.bank, split.arrays),
                       ConsensusFeatureExtractor.FEATURE_NAMES, g01_columns)
    g03 = take_columns(SymbolicVerificationFeatureExtractor().transform(split.bank, split.arrays),
                       SymbolicVerificationFeatureExtractor.FEATURE_NAMES, g03_columns)
    return np.concatenate([g01, g03], axis=-1).astype(np.float32), g01_columns, g03_columns


class ArmInput:
    """Dense part (always used as is) plus optional structured part standardized on the fit rows."""

    def __init__(self, name: str, dense_train: np.ndarray, dense_val: np.ndarray,
                 struct_train: Optional[np.ndarray] = None, struct_val: Optional[np.ndarray] = None,
                 side_train: Optional[np.ndarray] = None, side_val: Optional[np.ndarray] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.dense_train = dense_train
        self.dense_val = dense_val
        self.struct_train = struct_train
        self.struct_val = struct_val
        self.side_train = side_train
        self.side_val = side_val
        self.meta = meta or {}

    def build(self, fit_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        extra: Dict[str, np.ndarray] = {}
        if "centered_scale" in self.meta:
            extra["centered_scale"] = np.asarray([self.meta["centered_scale"]], dtype=np.float32)
        if self.struct_train is None:
            return self.dense_train, self.dense_val, extra
        mean, std = standardize_fit(self.struct_train[fit_rows])
        train = np.concatenate([self.dense_train, standardize_apply(self.struct_train, mean, std)], axis=-1)
        val = np.concatenate([self.dense_val, standardize_apply(self.struct_val, mean, std)], axis=-1)
        extra.update({
            "struct_mean": mean,
            "struct_std": std,
            "struct_column_count": np.asarray([self.struct_train.shape[-1]], dtype=np.int64),
        })
        return train.astype(np.float32), val.astype(np.float32), extra


# ---------------------------------------------------------------- decoding


def hits_of(chosen: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return labels[np.arange(labels.shape[0]), chosen].astype(bool)


def counts(chosen: np.ndarray, labels: np.ndarray, hard: np.ndarray) -> Dict[str, Any]:
    hits = hits_of(chosen, labels)
    return {"val_hard": accuracy_of(hits[hard]), "val_all": accuracy_of(hits)}


def decode(logits: np.ndarray, val: SplitData) -> Dict[str, Any]:
    """argmax and POST-P01/P02 with the EXP-B09 gamma grid and selection rule."""
    argmax_choice = np.argmax(logits, axis=1).astype(int)
    argmax = counts(argmax_choice, val.labels, val.hard)
    scaler = TemperatureScaler().fit(logits, val.labels)
    probs = scaler.transform(logits)
    rows = []
    for gamma in GAMMA_GRID:
        chosen = np.asarray(WeightedSelfConsistencyAggregator(gamma=gamma).aggregate(val.bank, probs), dtype=int)
        row = counts(chosen, val.labels, val.hard)
        row["gamma"] = gamma
        row["chosen"] = chosen
        rows.append(row)
    selected = select_gamma(rows, argmax["val_all"]["correct"])
    post_choice = selected["chosen"]
    sweep = [{k: v for k, v in row.items() if k != "chosen"} for row in rows]
    return {
        "argmax": argmax,
        "post": {
            "temperature": scaler.temperature,
            "gamma": selected["gamma"],
            "val_hard": selected["val_hard"],
            "val_all": selected["val_all"],
            "sweep": sweep,
        },
        "argmax_choice": argmax_choice,
        "post_choice": post_choice,
        "validation_configs_evaluated": 1 + len(GAMMA_GRID),
    }


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(v) for v in arr],
        "mean": float(arr.mean()),
        "std_ddof1": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def retention(new_hard: Sequence[int], ref_hard: Sequence[int], new_all: Sequence[int], ref_all: Sequence[int],
              paired: bool = True) -> Dict[str, Any]:
    """Pre-registered rule from the plan. Unpaired comparisons skip rule 2."""
    new_hard = np.asarray(new_hard, dtype=float)
    ref_hard = np.asarray(ref_hard, dtype=float)
    new_all = np.asarray(new_all, dtype=float)
    ref_all = np.asarray(ref_all, dtype=float)
    if paired:
        diffs = new_hard - ref_hard
        mean_gain = float(diffs.mean())
        positive = int((diffs > 0).sum())
        rule2 = positive >= 2
    else:
        diffs = None
        mean_gain = float(new_hard.mean() - ref_hard.mean())
        positive = None
        rule2 = True
    val_all_change = float(new_all.mean() - ref_all.mean())
    rule1 = mean_gain >= RETAIN_MIN_MEAN_GAIN
    rule3 = val_all_change >= -RETAIN_MAX_VAL_ALL_DROP
    return {
        "paired": paired,
        "val_hard_diffs": None if diffs is None else [float(d) for d in diffs],
        "mean_val_hard_gain_questions": mean_gain,
        "seeds_positive": positive,
        "mean_val_all_change_questions": val_all_change,
        "rule1_mean_gain_ge_3.55": rule1,
        "rule2_two_of_three_positive": rule2 if paired else "not applicable",
        "rule3_val_all_drop_le_4": rule3,
        "retained": bool(rule1 and rule2 and rule3),
    }


# ---------------------------------------------------------------- persistence


def save_npz(path: Path, weights: Dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_seeds(batch: str, entries: List[Dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = {
            "seed_set": list(SEEDS),
            "fold_partition_seed": FOLD_PARTITION_SEED,
            "derivation": {
                "full_model": "warmup = s, finetune = s + 1000",
                "fold_model": "warmup = 100*s + k, finetune = 100*s + k + 1000, k = 0..4",
                "tree": "random_state = s",
            },
            "batches": {},
        }
    registry["batches"][batch] = entries
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(WORKSPACE)).replace("\\", "/")


# ---------------------------------------------------------------- one arm, all seeds


def run_arm(
    arm: ArmInput,
    train: SplitData,
    val: SplitData,
    device: torch.device,
    out_dir: Path,
    with_folds: bool,
    module_factory: Optional[Callable[[int], torch.nn.Module]] = None,
    scorer: Callable = default_scorer,
) -> Dict[str, Any]:
    """Train the full model (and optional fold models) for every seed; return logits and records."""
    all_rows = np.arange(len(train.bank))
    hard_rows = all_rows[train.hard]
    folds = kfold_indices(len(train.bank), HPARAMS["n_folds"], FOLD_PARTITION_SEED) if with_folds else []
    per_seed = {}
    registry = []
    for seed in SEEDS:
        x_train, x_val, extra = arm.build(all_rows)
        seeds = full_model_seeds(seed)
        weights, stats = train_model(
            x_train, train.labels, x_train[hard_rows], train.labels[hard_rows], seeds, device,
            module_factory=module_factory,
            extra_warm=arm.side_train, extra_fine=None if arm.side_train is None else arm.side_train[hard_rows],
        )
        weights.update(extra)
        tick = time.perf_counter()
        val_logits = scorer(weights, x_val, arm.side_val)
        forward_seconds = time.perf_counter() - tick
        path = out_dir / f"seed_{seed}" / f"arm_{arm.name}_full.npz"
        digest = save_npz(path, weights)
        registry.append({"arm": arm.name, "seed": seed, "member": "full", "derived": seeds,
                         "file": rel(path), "sha256": digest})
        record = {
            "full": {
                "train_stats": stats,
                "uncompressed_bytes": uncompressed_weight_bytes(weights),
                "forward_seconds_val_all_numpy": forward_seconds,
                "sha256": digest,
                "file": rel(path),
            },
            "val_logits_full": val_logits,
        }
        print(f"  [{arm.name}] seed {seed} full: "
              f"argmax val_hard {counts(np.argmax(val_logits, 1), val.labels, val.hard)['val_hard']['correct']}"
              f"  ({stats['fit_seconds']:.1f}s)")
        if with_folds:
            fold_logits = []
            fold_rows = []
            for k, held in enumerate(folds):
                fit_rows = np.concatenate([folds[o] for o in range(len(folds)) if o != k])
                fit_hard = fit_rows[train.hard[fit_rows]]
                held_hard = held[train.hard[held]]
                xf_train, xf_val, extra_f = arm.build(fit_rows)
                fseeds = fold_model_seeds(seed, k)
                fweights, fstats = train_model(
                    xf_train[fit_rows], train.labels[fit_rows], xf_train[fit_hard], train.labels[fit_hard],
                    fseeds, device, module_factory=module_factory,
                    extra_warm=None if arm.side_train is None else arm.side_train[fit_rows],
                    extra_fine=None if arm.side_train is None else arm.side_train[fit_hard],
                )
                fweights.update(extra_f)
                held_scores = scorer(fweights, xf_train[held_hard],
                                     None if arm.side_train is None else arm.side_train[held_hard])
                held_hits = hits_of(np.argmax(held_scores, 1), train.labels[held_hard])
                fold_logits.append(scorer(fweights, xf_val, arm.side_val))
                fpath = out_dir / f"seed_{seed}" / f"arm_{arm.name}_fold{k}.npz"
                fdigest = save_npz(fpath, fweights)
                registry.append({"arm": arm.name, "seed": seed, "member": f"fold{k}", "derived": fseeds,
                                 "file": rel(fpath), "sha256": fdigest})
                fold_rows.append({"fold": k, "held_out_hard": accuracy_of(held_hits),
                                  "fit_seconds": fstats["fit_seconds"], "sha256": fdigest, "file": rel(fpath),
                                  "uncompressed_bytes": uncompressed_weight_bytes(fweights)})
            record["folds"] = fold_rows
            record["held_out_hard_mean"] = float(np.mean([r["held_out_hard"]["acc"] for r in fold_rows]))
            record["val_logits_fold_mean"] = np.mean(np.stack(fold_logits), axis=0)
            print(f"  [{arm.name}] seed {seed} folds: held-out hard mean {record['held_out_hard_mean'] * 100:.2f}%")
        per_seed[seed] = record
    return {"per_seed": per_seed, "registry": registry}


def save_logits(path: Path, logits: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **logits)
