"""EXP-B29 freeze, validation parity, and one-shot official test evaluation.

The original DASE7506-MP2 tree is only read. Every new file stays under dev/.
Test inputs and references stay unread until the freeze manifest has been sealed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import types
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "dev"
OFFICIAL = ROOT / "DASE7506-MP2"
CONTRACT_PATH = DEV / "runs" / "audit_logs" / "private_configuration_omitted.json"
THIRD_PATH = DEV / "runs" / "audit_logs" / "private_configuration_omitted.json"
SECOND_PATH = DEV / "runs" / "audit_logs" / "private_configuration_omitted.json"
OUT = DEV / "runs" / "exp_b29"
BUNDLE = DEV / "runs" / "model_bundle"
SOURCE_ASSETS = DEV / "runs" / "exp_b20" / "model_assets"
PY = Path(r"<user-home>\.conda\envs\mp1\python.exe")
PROTOCOL = "dase7506-mp2-open-hard-verifier-v5"
ASSET_LIMIT = 16 * 1024 * 1024
LIMIT_NS = 30 * 1_000_000_000
EXPECTED_PY = r"<user-home>\.conda\envs\mp1\python.exe"

RUNTIME = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "CUDA_VISIBLE_DEVICES": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
}


class FreezeFailure(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    body = {key: value for key, value in payload.items() if key not in {"freeze_id", "signature"}}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_official() -> List[Dict[str, Any]]:
    rows = []
    for dirpath, dirnames, filenames in os.walk(OFFICIAL):
        dirnames.sort()
        current = Path(dirpath)
        rows.append({
            "path": current.relative_to(OFFICIAL).as_posix(),
            "kind": "dir",
            "size": 0,
            "mtime_ns": current.stat().st_mtime_ns,
        })
        for name in sorted(filenames):
            path = current / name
            stat = path.stat()
            rows.append({
                "path": path.relative_to(OFFICIAL).as_posix(),
                "kind": "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    return rows


def prepare_dirs() -> None:
    for relative in (
        "tmp",
        "cache/matplotlib",
        "cache/torch",
        "cache/huggingface",
        "cache/pip",
        "cache/pycache",
        "cache/pytest",
        "package",
        "relocation/project",
        "mock_audit",
        "preseal",
        "primary",
        "reproduction",
        "asset_accounting",
        "environments",
        "logs",
    ):
        (OUT / relative).mkdir(parents=True, exist_ok=True)


def redirected_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(RUNTIME)
    env["TEMP"] = str((OUT / "tmp").resolve())
    env["TMP"] = str((OUT / "tmp").resolve())
    env["TMPDIR"] = str((OUT / "tmp").resolve())
    env["XDG_CACHE_HOME"] = str((OUT / "cache").resolve())
    env["MPLCONFIGDIR"] = str((OUT / "cache" / "matplotlib").resolve())
    env["TORCH_HOME"] = str((OUT / "cache" / "torch").resolve())
    env["HF_HOME"] = str((OUT / "cache" / "huggingface").resolve())
    env["PIP_CACHE_DIR"] = str((OUT / "cache" / "pip").resolve())
    env["PYTHONPYCACHEPREFIX"] = str((OUT / "cache" / "pycache").resolve())
    env.pop("PYTHONPATH", None)
    env.pop("B29_GUARD", None)
    return env


def guard_env(allowed_split: str, output: Path, allow_roots: Sequence[Path], log_name: str) -> Dict[str, str]:
    env = redirected_env()
    env["PYTHONPATH"] = str((DEV / "src" / "b29_sitecustomize").resolve())
    env["B29_GUARD"] = "1"
    env["B29_WORKSPACE"] = str(ROOT.resolve())
    env["B29_OFFICIAL_ROOT"] = str(OFFICIAL.resolve())
    env["B29_ALLOWED_SPLIT"] = allowed_split
    env["B29_OUTPUT_PATH"] = str(output.resolve())
    env["B29_ALLOW_ROOTS"] = ";".join(str(path.resolve()) for path in allow_roots)
    env["B29_DENY_LOG"] = str((OUT / "logs" / log_name).resolve())
    return env


def run_logged(cmd: Sequence[str], env: Dict[str, str], cwd: Path, log_dir: Path, timeout: int) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "command.json").write_text(json.dumps(list(cmd), indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (log_dir / "stdout.bin").write_bytes(exc.stdout or b"")
        (log_dir / "stderr.bin").write_bytes(exc.stderr or b"")
        raise FreezeFailure(f"process timed out after {timeout}s: {cmd}") from exc
    elapsed = time.perf_counter_ns() - started
    (log_dir / "stdout.bin").write_bytes(proc.stdout)
    (log_dir / "stderr.bin").write_bytes(proc.stderr)
    (log_dir / "exit.txt").write_text(str(proc.returncode), encoding="utf-8")
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise FreezeFailure(f"exit {proc.returncode}: {cmd}\n{tail}")
    return {"returncode": proc.returncode, "wall_ns": elapsed, "stdout": proc.stdout, "stderr": proc.stderr}


def require_guard(stderr: bytes) -> None:
    if b"B29_GUARD_ACTIVE" not in stderr:
        raise FreezeFailure("prediction process did not activate the file/network guard")


def load_common():
    official_code = str(OFFICIAL / "code")
    if official_code not in sys.path:
        sys.path.insert(0, official_code)
    import common  # type: ignore
    return common


def validation_bank_and_embeddings(common):
    data = OFFICIAL / "code" / "data"
    bank = common.read_jsonl(data / "candidates" / "gsm8k_validation.jsonl")
    gold = common.read_jsonl(data / "gsm8k_validation.jsonl")
    with np.load(data / "candidates" / "gsm8k_validation.npz", allow_pickle=False) as handle:
        embeddings = handle["embeddings"].copy()
    if embeddings.shape == (len(bank) * 4, 1024):
        embeddings = embeddings.reshape(len(bank), 4, 1024)
    if embeddings.shape != (len(bank), 4, 1024):
        raise FreezeFailure(f"unexpected validation embedding shape {embeddings.shape}")
    return bank, gold, embeddings


def hard_easy(common, bank, gold, selected: Sequence[int]) -> Dict[str, int]:
    if [row["id"] for row in bank] != [row["id"] for row in gold]:
        raise FreezeFailure("validation bank and reference ids differ")
    if len(selected) != len(bank):
        raise FreezeFailure("choice count differs from the validation bank")
    counts = {"hard_correct": 0, "hard_n": 0, "easy_correct": 0, "easy_n": 0}
    for row, gold_row, choice in zip(bank, gold, selected):
        parsed = [common.parse_answer(candidate["text"]) for candidate in row["candidates"]]
        distinct = {item for item in parsed if item is not None}
        hit = parsed[choice] is not None and parsed[choice] == common.parse_answer(gold_row["answer"])
        key = "hard" if len(distinct) >= 2 else "easy"
        counts[f"{key}_n"] += 1
        counts[f"{key}_correct"] += int(hit)
    return counts


def semantic_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: payload[key] for key in ("protocol", "split", "ids", "selected")}


def assert_prediction_match(actual: Dict[str, Any], expected: Dict[str, Any], label: str) -> None:
    if semantic_prediction(actual) != semantic_prediction(expected):
        raise FreezeFailure(f"{label} does not match the archived B20 validation choices")


def write_split_bank(data_dir: Path, split: str, rows: Sequence[Dict[str, Any]], embeddings: np.ndarray) -> None:
    candidates = data_dir / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    jsonl = candidates / f"gsm8k_{split}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    npz = candidates / f"gsm8k_{split}.npz"
    flat = np.ascontiguousarray(embeddings.reshape(len(rows) * 4, 1024))
    np.savez(npz, embeddings=flat)
    manifest = {
        "protocol": PROTOCOL,
        "splits": {split: len(rows)},
        "sha256": {
            f"candidates/gsm8k_{split}.jsonl": sha256_file(jsonl),
            f"candidates/gsm8k_{split}.npz": sha256_file(npz),
        },
    }
    write_json(data_dir / "manifest.json", manifest)


def predict_command(method: str, checkpoint: str, split: str, data_dir: str, output: str) -> List[str]:
    return [
        str(PY), "-B", "DASE7506-MP2/code/predict.py",
        "--method", method,
        "--checkpoint", checkpoint,
        "--split", split,
        "--data-dir", data_dir,
        "--output", output,
    ]


def score_command(split: str, predictions: str, output: str, data_dir: str = "DASE7506-MP2/code/data") -> List[str]:
    return [
        str(PY), "-B", "DASE7506-MP2/code/score_predictions.py",
        "--split", split,
        "--data-dir", data_dir,
        "--predictions", predictions,
        "--output", output,
    ]


def normal_allow_roots(output: Path, extra: Sequence[Path] = ()) -> List[Path]:
    roots = [
        OFFICIAL,
        DEV / "src",
        BUNDLE,
        OUT / "tmp",
        OUT / "cache",
        output.parent,
        *extra,
    ]
    return roots


def copy_exact(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)
    return sha256_bytes(data)


def numeric_lists(path: Path) -> List[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    class Visitor(ast.NodeVisitor):
        def visit_List(self, node):  # noqa: N802
            values = node.elts
            if len(values) > 8 and all(isinstance(item, ast.Constant) and isinstance(item.value, (int, float)) and not isinstance(item.value, bool) for item in values):
                found.append(len(values))
            self.generic_visit(node)

        def visit_Tuple(self, node):  # noqa: N802
            self.visit_List(node)

    Visitor().visit(tree)
    return found


def inventory_npz(path: Path) -> Dict[str, Any]:
    members = []
    expanded = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            expanded += info.file_size
            members.append({
                "member": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
            })
    payload = 0
    with np.load(path, allow_pickle=False) as handle:
        arrays = []
        for key in handle.files:
            array = handle[key]
            payload += int(array.nbytes)
            arrays.append({"name": key, "dtype": str(array.dtype), "shape": list(array.shape), "nbytes": int(array.nbytes)})
    disk = path.stat().st_size
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "format": "npz",
        "sha256": sha256_file(path),
        "disk_bytes": disk,
        "array_payload_bytes": payload,
        "expanded_bytes_including_headers": expanded,
        "members": members,
        "arrays": arrays,
        "decision_bytes": max(disk, expanded),
    }


def walk_arrays(obj, seen: set, arrays: List[np.ndarray]) -> None:
    marker = id(obj)
    if marker in seen:
        return
    seen.add(marker)
    if isinstance(obj, (str, bytes, int, float, bool, type(None), np.generic)):
        return
    if isinstance(obj, (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
        return
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            walk_arrays(key, seen, arrays)
            walk_arrays(value, seen, arrays)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            walk_arrays(value, seen, arrays)
        return
    if hasattr(obj, "__dict__"):
        walk_arrays(vars(obj), seen, arrays)
    slots = getattr(obj, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for name in slots or ():
        try:
            walk_arrays(getattr(obj, name), seen, arrays)
        except Exception:
            continue


def inventory_pickle(path: Path) -> Dict[str, Any]:
    import pickle
    import sklearn.ensemble  # noqa: F401

    with path.open("rb") as handle:
        model = pickle.load(handle)
    arrays: List[np.ndarray] = []
    walk_arrays(model, set(), arrays)
    payload = int(sum(array.nbytes for array in arrays))
    disk = path.stat().st_size
    canonical = None
    decision = disk
    if payload > disk:
        dump = OUT / "asset_accounting" / "tree_r5"
        dump.mkdir(parents=True, exist_ok=True)
        total = 0
        manifest_rows = []
        for index, array in enumerate(arrays):
            target = dump / f"array_{index}.npy"
            np.save(target, array)
            size = target.stat().st_size
            total += size
            manifest_rows.append({"file": target.name, "bytes": size, "shape": list(array.shape), "dtype": str(array.dtype)})
        state = {"reconstructed_array_nbytes": payload, "arrays": manifest_rows}
        state_path = dump / "state.json"
        write_json(state_path, state)
        total += state_path.stat().st_size
        canonical = total
        decision = max(disk, total)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "format": "pickle",
        "sha256": sha256_file(path),
        "disk_bytes": disk,
        "array_payload_bytes": payload,
        "expanded_bytes_including_headers": disk,
        "canonical_reconstructed_state_bytes_if_needed": canonical,
        "decision_bytes": decision,
        "estimator": type(model).__name__,
    }


def inventory_raw(path: Path) -> Dict[str, Any]:
    disk = path.stat().st_size
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "format": "raw",
        "sha256": sha256_file(path),
        "disk_bytes": disk,
        "array_payload_bytes": 0,
        "expanded_bytes_including_headers": disk,
        "decision_bytes": disk,
    }


def package_environment() -> Dict[str, Any]:
    rows = []
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
            wheel = dist.read_text("WHEEL") or ""
            tags = [line.split(":", 1)[1].strip() for line in wheel.splitlines() if line.startswith("Tag:")]
        except Exception as exc:  # metadata-only inventory
            rows.append({"error": str(exc)})
            continue
        rows.append({"name": name, "version": version, "wheel_tags": tags})
    rows.sort(key=lambda item: item.get("name", "").lower())
    lines = [
        "# Existing mp1 environment inventory recorded for EXP-B29.",
        "# This is not a clean-room reinstall and B29 does not modify that environment.",
    ]
    lines.extend(f"{item['name']}=={item['version']}" for item in rows if "name" in item)
    lock_path = DEV / "requirements-freeze.txt"
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "interpreter": str(PY),
        "version": sys.version,
        "platform": sys.platform,
        "clean_install_verified": False,
        "distributions": rows,
    }
    write_json(OUT / "environment-freeze.json", payload)
    return {"lock_sha256": sha256_file(lock_path), "environment_sha256": sha256_file(OUT / "environment-freeze.json"), "distribution_count": len(rows)}


def powershell(command: str) -> str:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def hardware_record() -> Dict[str, Any]:
    return {
        "cpu_name": powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"),
        "physical_cores": powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty NumberOfCores)"),
        "logical_processors": powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty NumberOfLogicalProcessors)"),
        "ram_bytes": powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
        "os": powershell("[Environment]::OSVersion.VersionString"),
        "architecture": os.environ.get("PROCESSOR_ARCHITECTURE", ""),
        "python_cpu_count": os.cpu_count(),
        "affinity_restricted": False,
        "requested_numeric_threads": 1,
        "gpu_hidden_by_env": "CUDA_VISIBLE_DEVICES=-1",
    }


def hash_manifest_files(contract: Dict[str, Any], include_test: bool) -> Dict[str, Any]:
    manifest = read_json(OFFICIAL / "code" / "PACKAGE_MANIFEST.json")
    rows = []
    for relative, expected in manifest["files"].items():
        is_test = relative.endswith("gsm8k_test.jsonl") or relative.endswith("gsm8k_test.npz")
        if is_test and not include_test:
            rows.append({"file": relative, "skipped_until_seal": True})
            continue
        path = OFFICIAL / relative
        actual = sha256_file(path)
        rows.append({"file": relative, "expected": expected, "actual": actual, "matches": actual == expected})
        if actual != expected:
            raise FreezeFailure(f"official hash mismatch: {relative}")
    return {"include_test": include_test, "files": rows}


def populate_bundle(contract: Dict[str, Any]) -> None:
    expected = {item["file"]: item["sha256"] for item in contract["locked_asset_inventory"]}
    if set(expected) != {
        "arm_a_seed_42.npz", "arm_a_seed_43.npz", "arm_a_seed_44.npz",
        "arm_b_seed_42.npz", "arm_b_seed_43.npz", "arm_b_seed_44.npz",
        "config.json", "svd_components.npz", "tree_r5.pkl",
    }:
        raise FreezeFailure("locked asset inventory is not the nine B20 files")
    existing = {path.name for path in BUNDLE.iterdir()}
    allowed_existing = set(expected) | {".gitkeep"}
    if not existing <= allowed_existing:
        raise FreezeFailure(f"model_bundle has unexpected files: {sorted(existing - allowed_existing)}")
    gitkeep = BUNDLE / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()
    for name, digest in expected.items():
        src = SOURCE_ASSETS / name
        actual = sha256_file(src)
        if actual != digest:
            raise FreezeFailure(f"source asset hash changed: {name}")
        target = BUNDLE / name
        if target.exists() and sha256_file(target) == digest:
            continue
        copy_exact(src, target)
        if sha256_file(target) != digest:
            raise FreezeFailure(f"copied asset hash changed: {name}")
    final_names = {path.name for path in BUNDLE.iterdir()}
    if final_names != set(expected):
        raise FreezeFailure(f"bundle contents are not exactly the nine assets: {sorted(final_names)}")


def build_asset_inventory() -> Dict[str, Any]:
    files = []
    embedded = []
    for path in sorted(BUNDLE.iterdir()):
        if path.suffix == ".npz":
            files.append(inventory_npz(path))
        elif path.suffix == ".pkl":
            files.append(inventory_pickle(path))
        else:
            files.append(inventory_raw(path))
    for path in (DEV / "src").rglob("*.py"):
        if "b29" in path.name or path.name == "submission.py" or "features" in path.parts or "models" in path.parts or "postprocessing" in path.parts:
            found = numeric_lists(path)
            if found:
                embedded.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "numeric_list_lengths": found})
    if embedded:
        raise FreezeFailure(f"embedded numeric lists require accounting: {embedded}")
    total = int(sum(item["decision_bytes"] for item in files))
    payload = {
        "files": files,
        "embedded_learned_literals": embedded,
        "decision_bytes": total,
        "limit_bytes": ASSET_LIMIT,
        "passed": total <= ASSET_LIMIT,
        "runtime_policy_read_by_prediction": False,
        "note": "Guard rules live in ordinary source. freeze_manifest.json is controller-only and is not read by predict().",
    }
    if not payload["passed"]:
        raise FreezeFailure(f"asset decision bytes {total} exceed {ASSET_LIMIT}")
    write_json(OUT / "asset_inventory.json", payload)
    return payload


def run_predict(split: str, output: Path, log_dir: Path, extra_roots: Sequence[Path] = (), cwd: Path = ROOT, command: Optional[List[str]] = None, pythonpath: Optional[Path] = None, allow_roots: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if command is None:
        command = predict_command(
            "dev/src/submission.py",
            "dev/runs/model_bundle",
            split,
            "DASE7506-MP2/code/data",
            str(output.relative_to(cwd)).replace("\\", "/"),
        )
    roots = list(allow_roots) if allow_roots is not None else normal_allow_roots(output, extra_roots)
    env = guard_env(split, output, roots, f"{log_dir.name}-guard.log")
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath.resolve())
    result = run_logged(command, env, cwd, log_dir, timeout=180)
    require_guard(result["stderr"])
    return result


def official_score(split: str, predictions: Path, output: Path, log_dir: Path) -> Dict[str, Any]:
    command = score_command(split, str(predictions.relative_to(ROOT)).replace("\\", "/"), str(output.relative_to(ROOT)).replace("\\", "/"))
    result = run_logged(command, redirected_env(), ROOT, log_dir, timeout=180)
    output_payload = read_json(output)
    stdout_payload = json.loads(result["stdout"].decode("utf-8"))
    if {key: output_payload[key] for key in ("protocol", "split", "correct", "n", "accuracy")} != {
        key: stdout_payload[key] for key in ("protocol", "split", "correct", "n", "accuracy")
    }:
        raise FreezeFailure("scorer stdout and output file disagree")
    return output_payload


def timed_official_pair(split: str, pred_output: Path, score_output: Path, pred_cmd: List[str], score_cmd: List[str], log_stem: str) -> Dict[str, Any]:
    pred_output.parent.mkdir(parents=True, exist_ok=True)
    env = guard_env(split, pred_output, normal_allow_roots(pred_output), f"{log_stem}-guard.log")
    continuous_start = time.perf_counter_ns()
    prediction = run_logged(pred_cmd, env, ROOT, OUT / "logs" / f"{log_stem}-predict", timeout=180)
    require_guard(prediction["stderr"])
    prediction_hash = sha256_file(pred_output)
    score = run_logged(score_cmd, redirected_env(), ROOT, OUT / "logs" / f"{log_stem}-score", timeout=180)
    continuous_ns = time.perf_counter_ns() - continuous_start
    pred_stdout = json.loads(prediction["stdout"].decode("utf-8"))
    score_payload = read_json(score_output)
    score_stdout = json.loads(score["stdout"].decode("utf-8"))
    summary = {
        "predict_wall_ns": prediction["wall_ns"],
        "score_wall_ns": score["wall_ns"],
        "sum_process_wall_ns": prediction["wall_ns"] + score["wall_ns"],
        "continuous_pair_wall_ns": continuous_ns,
        "parent_gap_ns": continuous_ns - prediction["wall_ns"] - score["wall_ns"],
        "runner_prediction_seconds": pred_stdout.get("prediction_seconds"),
        "runner_scoring_seconds": score_stdout.get("scoring_seconds"),
        "exit_codes": [prediction["returncode"], score["returncode"]],
        "prediction_sha256": prediction_hash,
        "within_limit": prediction["wall_ns"] + score["wall_ns"] <= LIMIT_NS and continuous_ns <= LIMIT_NS,
    }
    if {key: score_payload[key] for key in ("correct", "n", "accuracy")} != {key: score_stdout[key] for key in ("correct", "n", "accuracy")}:
        raise FreezeFailure("timed scorer outputs disagree")
    return summary


def direct_choices(data_dir: Path, split: str, checkpoint: Path, output: Path, log_name: str, pythonpath: Optional[Path] = None, script: Optional[Path] = None, allow_roots: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    script = script or (DEV / "src" / "experiments" / "exp_b29_direct_predict.py")
    command = [str(PY), "-B", str(script), "--checkpoint", str(checkpoint), "--data-dir", str(data_dir), "--split", split, "--output", str(output)]
    roots = list(allow_roots) if allow_roots is not None else normal_allow_roots(output, [data_dir])
    env = guard_env(split, output, roots, f"{log_name}-guard.log")
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath.resolve())
    run_logged(command, env, ROOT, OUT / "logs" / log_name, timeout=180)
    return read_json(output)


def make_audit_banks(common, bank, embeddings) -> Dict[str, Any]:
    rng = np.random.default_rng(29029)
    reports = []
    saved = {}
    for size in (1, 17, 137):
        picked = rng.choice(len(bank), size=size, replace=False)
        subset_rows = [json.loads(json.dumps(bank[int(index)])) for index in picked]
        subset_emb = embeddings[picked]
        order = rng.permutation(size)
        perm_rows = [json.loads(json.dumps(subset_rows[int(index)])) for index in order]
        perm_emb = subset_emb[order]
        for position, row in enumerate(perm_rows):
            row["id"] = f"b29-audit-{size}-{position}"
        data_dir = OUT / "mock_audit" / f"n{size}" / "data"
        write_split_bank(data_dir, "audit", perm_rows, perm_emb)
        output = OUT / "mock_audit" / f"n{size}" / "audit-predictions.json"
        command = [
            str(PY), "-B", str(OFFICIAL / "code" / "predict.py"),
            "--method", str(DEV / "src" / "submission.py"),
            "--checkpoint", str(BUNDLE),
            "--split", "audit",
            "--data-dir", str(data_dir),
            "--output", str(output),
        ]
        run_predict("audit", output, OUT / "logs" / f"audit-{size}", extra_roots=[data_dir], command=command)
        runner = read_json(output)
        direct_dir = OUT / "mock_audit" / f"n{size}" / "direct_data"
        write_split_bank(direct_dir, "audit", subset_rows, subset_emb)
        direct = direct_choices(direct_dir, "audit", BUNDLE, OUT / "mock_audit" / f"n{size}" / "direct-choices.json", f"audit-direct-{size}")
        undone = [None] * size
        for new_pos, old_pos in enumerate(order):
            undone[int(old_pos)] = runner["selected"][new_pos]
        if undone != direct["selected"]:
            raise FreezeFailure(f"audit bank n={size} does not match direct recomputation")
        if any(type(choice) is not int or not 0 <= choice < 4 for choice in runner["selected"]):
            raise FreezeFailure(f"audit bank n={size} produced a non-Python-int choice")
        reports.append({"size": size, "ids": runner["ids"], "selected": runner["selected"], "direct_match": True})
        saved[size] = {"rows": perm_rows, "embeddings": perm_emb, "selected": runner["selected"], "data_dir": data_dir}
    content_rows = [json.loads(json.dumps(row)) for row in saved[17]["rows"]]
    content_emb = saved[17]["embeddings"].copy()
    content_emb[0, 0, 0] = np.float32(content_emb[0, 0, 0] + 3.0)
    content_rows[0]["candidates"][0]["text"] = content_rows[0]["candidates"][0]["text"] + "\n#### 999999"
    content_dir = OUT / "mock_audit" / "content" / "data"
    write_split_bank(content_dir, "audit", content_rows, content_emb)
    content_out = OUT / "mock_audit" / "content" / "audit-predictions.json"
    command = [
        str(PY), "-B", str(OFFICIAL / "code" / "predict.py"),
        "--method", str(DEV / "src" / "submission.py"),
        "--checkpoint", str(BUNDLE),
        "--split", "audit",
        "--data-dir", str(content_dir),
        "--output", str(content_out),
    ]
    run_predict("audit", content_out, OUT / "logs" / "audit-content", extra_roots=[content_dir], command=command)
    content_runner = read_json(content_out)
    content_direct = direct_choices(content_dir, "audit", BUNDLE, OUT / "mock_audit" / "content" / "direct-choices.json", "audit-content-direct")
    if content_runner["selected"] != content_direct["selected"] or content_runner["ids"] != content_direct["ids"]:
        raise FreezeFailure("modified-content audit rerun disagrees with direct recomputation")
    return {
        "seed": 29029,
        "sizes": reports,
        "content_sensitivity": {
            "ids_reused": True,
            "runner_matches_direct": True,
            "choices_changed_from_unmodified": content_runner["selected"] != saved[17]["selected"],
            "answer_change_required": False,
        },
    }


def copy_relocation(closure_files: Sequence[Path], audit_data: Path) -> Path:
    project = OUT / "relocation" / "project"
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    for relative in ("code/predict.py", "code/common.py", "code/score_predictions.py"):
        copy_exact(OFFICIAL / relative, project / "DASE7506-MP2" / relative)
    data_src = OFFICIAL / "code" / "data"
    for relative in (
        "manifest.json",
        "candidates/gsm8k_validation.jsonl",
        "candidates/gsm8k_validation.npz",
    ):
        copy_exact(data_src / relative, project / "DASE7506-MP2" / "code" / "data" / relative)
    for src in closure_files:
        copy_exact(src, project / src.relative_to(ROOT))
    for name in sorted(path.name for path in BUNDLE.iterdir()):
        copy_exact(BUNDLE / name, project / "dev" / "runs" / "model_bundle" / name)
    relocated_audit = project / "audit_data"
    for path in audit_data.rglob("*"):
        if path.is_file():
            copy_exact(path, relocated_audit / path.relative_to(audit_data))
    return project


def run_relocation(project: Path, reference_prediction: Dict[str, Any], audit_selected: Sequence[int]) -> Dict[str, Any]:
    pythonpath = project / "dev" / "src" / "b29_sitecustomize"
    allow = [project, OUT / "tmp", OUT / "cache", OUT / "relocation"]
    validation_out = OUT / "relocation" / "validation-predictions.json"
    command = [
        str(PY), "-B", str(project / "DASE7506-MP2" / "code" / "predict.py"),
        "--method", str(project / "dev" / "src" / "submission.py"),
        "--checkpoint", str(project / "dev" / "runs" / "model_bundle"),
        "--split", "validation",
        "--data-dir", str(project / "DASE7506-MP2" / "code" / "data"),
        "--output", str(validation_out),
    ]
    run_predict(
        "validation",
        validation_out,
        OUT / "logs" / "relocation-validation",
        command=command,
        pythonpath=pythonpath,
        allow_roots=allow,
        cwd=project,
    )
    relocated = read_json(validation_out)
    assert_prediction_match(relocated, reference_prediction, "relocated validation")
    audit_out = OUT / "relocation" / "audit-predictions.json"
    audit_cmd = [
        str(PY), "-B", str(project / "DASE7506-MP2" / "code" / "predict.py"),
        "--method", str(project / "dev" / "src" / "submission.py"),
        "--checkpoint", str(project / "dev" / "runs" / "model_bundle"),
        "--split", "audit",
        "--data-dir", str(project / "audit_data"),
        "--output", str(audit_out),
    ]
    run_predict("audit", audit_out, OUT / "logs" / "relocation-audit", command=audit_cmd, pythonpath=pythonpath, allow_roots=allow, cwd=project)
    audit_payload = read_json(audit_out)
    if audit_payload["selected"] != list(audit_selected):
        raise FreezeFailure("relocated audit choices differ from the sealed-source audit run")
    probe_cmd = [
        str(PY), "-B", str(project / "dev" / "src" / "experiments" / "exp_b29_probe.py"),
        "--kind", "path",
        "--path", str(DEV / "src" / "submission.py"),
    ]
    probe_env = guard_env("audit", audit_out, allow, "relocation-probe-guard.log")
    probe_env["PYTHONPATH"] = str(pythonpath.resolve())
    probe = run_logged(probe_cmd, probe_env, project, OUT / "logs" / "relocation-probe", timeout=60)
    require_guard(probe["stderr"])
    if b'"denied": true' not in probe["stdout"].lower():
        raise FreezeFailure("relocation process was able to read the original submission.py")
    return {"validation_match": True, "audit_match": True, "original_source_denied": True}


def collect_closure() -> Dict[str, Any]:
    output = OUT / "preseal" / "closure-report.json"
    command = [
        str(PY), "-B", str(DEV / "src" / "experiments" / "exp_b29_probe.py"),
        "--kind", "closure",
        "--checkpoint", str(BUNDLE),
        "--output", str(output),
    ]
    result = run_logged(
        command,
        guard_env("validation", output, normal_allow_roots(output), "closure-guard.log"),
        ROOT,
        OUT / "logs" / "closure",
        timeout=180,
    )
    require_guard(result["stderr"])
    report = read_json(output)
    if report["cuda_available"]:
        raise FreezeFailure("CUDA is visible inside the sealed prediction environment")
    if report["torch_threads"] != 1 or report["torch_interop_threads"] != 1:
        raise FreezeFailure(f"unexpected torch threads: {report['torch_threads']}/{report['torch_interop_threads']}")
    src_files = []
    for item in report["modules"]:
        path = Path(item["file"])
        try:
            path.relative_to(DEV / "src")
        except ValueError:
            continue
        if path.suffix == ".py":
            src_files.append(path)
    extra = [
        DEV / "src" / "b29_guard.py",
        DEV / "src" / "b29_sitecustomize" / "sitecustomize.py",
        DEV / "src" / "experiments" / "exp_b29_direct_predict.py",
        DEV / "src" / "experiments" / "exp_b29_probe.py",
        DEV / "src" / "submission.py",
    ]
    unique = []
    seen = set()
    for path in [*src_files, *extra]:
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            unique.append(resolved)
    return {"report": report, "files": unique}


def preseal_hashes(contract: Dict[str, Any]) -> None:
    snapshot = contract["preparation_snapshot"]["source_sha256"]
    allowed_change = {"dev/src/submission.py"}
    for relative, expected in snapshot.items():
        actual = sha256_file(ROOT / relative)
        if actual == expected:
            continue
        if relative in allowed_change:
            continue
        raise FreezeFailure(f"unexpected source drift: {relative}")
    for relative, expected in contract["predecessor_contract_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise FreezeFailure(f"predecessor contract drift: {relative}")
    plan_hash = sha256_file(DEV / "private_document_omitted")
    if plan_hash != contract["planning_verification"]["reviewed_plan_sha256"]:
        raise FreezeFailure("private_document_omitted hash differs from the freeze contract")
    third = read_json(THIRD_PATH)
    expected_status = {
        "EXP-B25": "validation_not_retained",
        "EXP-B26": "stopped_internal_gate",
        "EXP-B27": "validation_not_retained",
        "EXP-B28": "skipped_because_b26_internal_gate_failed",
    }
    for batch, status in expected_status.items():
        if third["batches"][batch]["status"] != status:
            raise FreezeFailure(f"{batch} disposition changed")
    if "B20" not in contract["method_lock"]["selected"]:
        raise FreezeFailure("method lock is not historical B20")


def run_validation_parity(common, reference: Dict[str, Any], bank, gold) -> Dict[str, Any]:
    original_out = OUT / "preseal" / "original-validation-predictions.json"
    original_cmd = predict_command(
        "dev/runs/exp_b20/candidate_submission.py",
        "dev/runs/exp_b20/model_assets",
        "validation",
        "DASE7506-MP2/code/data",
        "dev/runs/exp_b29/preseal/original-validation-predictions.json",
    )
    original = run_logged(original_cmd, redirected_env(), ROOT, OUT / "logs" / "original-replay", timeout=180)
    if b"B29_GUARD_ACTIVE" in original["stderr"]:
        raise FreezeFailure("original replay unexpectedly installed the sealed guard")
    original_payload = read_json(original_out)
    assert_prediction_match(original_payload, reference, "original candidate replay")
    runs = []
    for index in (1, 2):
        output = OUT / "preseal" / f"final-run{index}" / "validation-predictions.json"
        score_out = OUT / "preseal" / f"final-run{index}" / "validation-score.json"
        command = predict_command(
            "dev/src/submission.py",
            "dev/runs/model_bundle",
            "validation",
            "DASE7506-MP2/code/data",
            str(output.relative_to(ROOT)).replace("\\", "/"),
        )
        run_predict("validation", output, OUT / "logs" / f"final-validation-{index}", command=command)
        payload = read_json(output)
        assert_prediction_match(payload, reference, f"final validation run {index}")
        score = official_score("validation", output, score_out, OUT / "logs" / f"final-validation-score-{index}")
        counts = hard_easy(common, bank, gold, payload["selected"])
        if score["correct"] != 743 or score["n"] != 798:
            raise FreezeFailure(f"official validation score is {score['correct']}/{score['n']}")
        if counts["hard_correct"] != 300 or counts["hard_n"] != 355:
            raise FreezeFailure(f"validation hard score is {counts}")
        if counts["easy_correct"] != 443 or counts["easy_n"] != 443:
            raise FreezeFailure(f"validation easy score is {counts}")
        runs.append({"official": {key: score[key] for key in ("correct", "n", "accuracy")}, "counts": counts})
    return {"runs": runs, "original_replay_match": True}


def seal_manifest(contract_hash: str, evidence: Dict[str, Any], assets: Dict[str, Any], environment: Dict[str, Any], closure_files: Sequence[Path]) -> Dict[str, Any]:
    manifest = {
        "experiment": "EXP-B29",
        "method": "EXP-B17/EXP-B20 historical three-view Soft Stacking",
        "sealed_utc": utc_now(),
        "contract_path": "dev/runs/audit_logs/private_configuration_omitted.json",
        "contract_sha256": contract_hash,
        "sealed_test_status": "SEALED_TEST_NOT_RUN",
        "runtime_profile": {
            "id": "conservative_serial_under_four_thread_cap",
            "requested_numeric_threads": 1,
            "environment": RUNTIME,
        },
        "commands": {
            "primary_prediction": predict_command("dev/src/submission.py", "dev/runs/model_bundle", "test", "DASE7506-MP2/code/data", "dev/runs/exp_b29/primary/test-predictions.json"),
            "primary_scoring": score_command("test", "dev/runs/exp_b29/primary/test-predictions.json", "dev/runs/exp_b29/primary/test-score.json"),
            "reproduction_prediction": predict_command("dev/src/submission.py", "dev/runs/model_bundle", "test", "DASE7506-MP2/code/data", "dev/runs/exp_b29/reproduction/test-predictions.json"),
            "reproduction_scoring": score_command("test", "dev/runs/exp_b29/reproduction/test-predictions.json", "dev/runs/exp_b29/reproduction/test-score.json"),
        },
        "assets": [
            {"path": item["path"], "sha256": item["sha256"], "decision_bytes": item["decision_bytes"]}
            for item in assets["files"]
        ],
        "asset_decision_bytes": assets["decision_bytes"],
        "source_closure": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path)}
            for path in closure_files
        ],
        "dependency_lock_sha256": environment["lock_sha256"],
        "environment_sha256": environment["environment_sha256"],
        "preseal": evidence,
        "provenance": {
            "weights": [0.6519943983184848, 0.2773940600630942, 0.07061154161842091],
            "temperature": 0.5,
            "mask_penalty": 1000000000.0,
            "members": "A/B seeds 42,43,44 and one R5 tree plus one SVD",
            "rejected_alternatives": ["B20 strict refit", "B25 decoder", "B26 centroid", "B27 energy", "B28 joint"],
        },
    }
    manifest["freeze_id"] = sha256_bytes(canonical_bytes(manifest))
    target = OUT / "freeze_manifest.json"
    write_json(target, manifest)
    loaded = read_json(target)
    if sha256_bytes(canonical_bytes(loaded)) != loaded["freeze_id"]:
        raise FreezeFailure("freeze_id does not match the canonical manifest payload")
    return loaded


def write_summary(state: Dict[str, Any]) -> None:
    payload = {
        "experiment": "EXP-B29",
        "finished_utc": utc_now(),
        "phase": state.get("phase"),
        "overall": state.get("overall"),
        "error": state.get("error"),
        "freeze_id": state.get("freeze_id"),
        "sealed_utc": state.get("sealed_utc"),
        "test_accuracy_percent": state.get("test_accuracy_percent"),
        "test_correct": state.get("test_correct"),
        "test_n": state.get("test_n"),
        "primary_timing": state.get("primary_timing"),
        "reproduction_timing": state.get("reproduction_timing"),
        "asset_decision_bytes": state.get("asset_decision_bytes"),
        "offline_evidence_level": state.get("offline_evidence_level"),
        "local_resource_verdict": state.get("local_resource_verdict"),
        "reference_machine_verdict": "PENDING_NOT_RUN",
        "reproducibility_verdict": state.get("reproducibility_verdict"),
        "test_inputs_opened_by_controller_hash": state.get("test_inputs_opened", False),
        "test_references_opened_by_official_tools": state.get("test_references_opened", False),
        "controller_parsed_test_labels": False,
        "written_outside_dev": False,
        "official_repository_unchanged": state.get("official_repository_unchanged"),
        "traceback": state.get("traceback"),
    }
    write_json(DEV / "runs" / "audit_logs" / "exp_b29_freeze_audit.json", payload)


def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    state["phase"] = "P0"
    state["overall"] = "PRESEAL_FAILED"
    state["test_references_opened"] = False
    state["test_inputs_opened"] = False
    if (OUT / "primary" / "test-predictions.json").exists() or (OUT / "reproduction" / "test-predictions.json").exists():
        raise FreezeFailure("test predictions already exist; refusing to rerun after test exposure")
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(Path(EXPECTED_PY).resolve())):
        raise FreezeFailure(f"interpreter is {sys.executable}")
    prepare_dirs()
    log("P0 snapshot official repository")
    official_snapshot = snapshot_official()
    state["official_snapshot"] = official_snapshot
    contract = read_json(CONTRACT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    preseal_hashes(contract)
    hash_manifest_files(contract, include_test=False)
    hardware = hardware_record()
    write_json(OUT / "repository_readonly_audit.json", {
        "started_utc": utc_now(),
        "official_entries": len(official_snapshot),
        "test_file_contents_read": False,
        "snapshot": official_snapshot,
    })

    log("P1 package assets and environment lock")
    state["phase"] = "P1"
    populate_bundle(contract)
    environment = package_environment()
    write_json(OUT / "package" / "commands.json", {
        "interpreter": str(PY),
        "cwd": str(ROOT),
        "primary_prediction": predict_command("dev/src/submission.py", "dev/runs/model_bundle", "test", "DASE7506-MP2/code/data", "dev/runs/exp_b29/primary/test-predictions.json"),
        "primary_scoring": score_command("test", "dev/runs/exp_b29/primary/test-predictions.json", "dev/runs/exp_b29/primary/test-score.json"),
        "note": "Commands recorded before test execution.",
    })

    log("P1 probes")
    synthetic = OUT / "tmp" / "synthetic_forbidden" / "gsm8k_test.jsonl"
    synthetic.parent.mkdir(parents=True, exist_ok=True)
    synthetic.write_text("synthetic\n", encoding="utf-8")
    probe_env = guard_env("validation", OUT / "tmp" / "probe-output.json", normal_allow_roots(OUT / "tmp" / "probe-output.json"), "probe-guard.log")
    reference_probe = run_logged(
        [str(PY), "-B", str(DEV / "src" / "experiments" / "exp_b29_probe.py"), "--kind", "reference", "--path", str(synthetic)],
        probe_env,
        ROOT,
        OUT / "logs" / "probe-reference",
        timeout=60,
    )
    require_guard(reference_probe["stderr"])
    network_probe = run_logged(
        [str(PY), "-B", str(DEV / "src" / "experiments" / "exp_b29_probe.py"), "--kind", "network"],
        probe_env,
        ROOT,
        OUT / "logs" / "probe-network",
        timeout=60,
    )
    require_guard(network_probe["stderr"])
    closure = collect_closure()
    assets = build_asset_inventory()
    state["asset_decision_bytes"] = assets["decision_bytes"]

    log("P2 validation parity")
    state["phase"] = "P2"
    common = load_common()
    reference = read_json(DEV / "runs" / "exp_b20" / "validation-predictions.json")
    bank, gold, embeddings = validation_bank_and_embeddings(common)
    archived_counts = hard_easy(common, bank, gold, reference["selected"])
    if archived_counts["hard_correct"] != 300 or archived_counts["hard_n"] != 355:
        raise FreezeFailure(f"archived B20 choices do not score 300/355 under the hard definition: {archived_counts}")
    parity = run_validation_parity(common, reference, bank, gold)
    log("P2 audit banks")
    audit = make_audit_banks(common, bank, embeddings)
    write_json(OUT / "audit_bank_checks.json", audit)
    log("P2 relocation")
    project = copy_relocation(closure["files"], OUT / "mock_audit" / "n137" / "data")
    relocation = run_relocation(project, reference, audit["sizes"][-1]["selected"])
    write_json(OUT / "validation_parity.json", {"parity": parity, "archived_counts": archived_counts, "relocation": relocation})
    offline = {
        "evidence_level": "software_guarded_offline_with_native_IO_review",
        "os_firewall_modified": False,
        "synthetic_reference_denied": True,
        "synthetic_network_denied": True,
        "method_network_attempts": 0,
        "native_io_note": "Python audit hooks cover open, socket, urllib and subprocess. Native library calls that bypass sys.audit are not an OS airgap. Prediction source does not call requests, urllib or socket.",
        "cuda_available": False,
        "threadpools": closure["report"]["threadpools"],
    }
    write_json(OUT / "offline_access_audit.json", offline)
    state["offline_evidence_level"] = offline["evidence_level"]
    if snapshot_official() != official_snapshot:
        state["official_repository_unchanged"] = False
        raise FreezeFailure("official repository changed before seal")

    log("P3 seal")
    state["phase"] = "P3"
    evidence = {
        "validation_official_correct": 743,
        "validation_hard_correct": 300,
        "validation_runs": 2,
        "audit_sizes": [1, 17, 137],
        "relocation_validation_match": True,
        "evidence_sha256": {
            "validation_parity": sha256_file(OUT / "validation_parity.json"),
            "audit_bank": sha256_file(OUT / "audit_bank_checks.json"),
            "offline": sha256_file(OUT / "offline_access_audit.json"),
            "assets": sha256_file(OUT / "asset_inventory.json"),
            "closure": sha256_file(OUT / "preseal" / "closure-report.json"),
        },
    }
    manifest = seal_manifest(contract_hash, evidence, assets, environment, closure["files"])
    manifest_hash = sha256_file(OUT / "freeze_manifest.json")
    state["freeze_id"] = manifest["freeze_id"]
    state["sealed_utc"] = manifest["sealed_utc"]
    state["overall"] = "SEALED_TEST_PENDING"

    log("P4 official integrity")
    state["phase"] = "P4"
    state["test_inputs_opened"] = True
    state["test_references_opened"] = True
    integrity_hashes = hash_manifest_files(contract, include_test=True)
    checker = run_logged(
        [str(PY), "-B", "DASE7506-MP2/code/check_data.py", "--data-dir", "DASE7506-MP2/code/data"],
        redirected_env(),
        ROOT,
        OUT / "logs" / "check-data",
        timeout=180,
    )
    integrity = {"hashes": integrity_hashes, "checker_stdout": json.loads(checker["stdout"].decode("utf-8")), "exit_code": checker["returncode"]}
    write_json(OUT / "official_integrity.json", integrity)
    if sha256_file(OUT / "freeze_manifest.json") != manifest_hash:
        raise FreezeFailure("freeze manifest changed during integrity checks")

    log("P5 primary test")
    state["phase"] = "P5"
    primary_pred = predict_command("dev/src/submission.py", "dev/runs/model_bundle", "test", "DASE7506-MP2/code/data", "dev/runs/exp_b29/primary/test-predictions.json")
    primary_score_cmd = score_command("test", "dev/runs/exp_b29/primary/test-predictions.json", "dev/runs/exp_b29/primary/test-score.json")
    primary_timing = timed_official_pair(
        "test",
        OUT / "primary" / "test-predictions.json",
        OUT / "primary" / "test-score.json",
        primary_pred,
        primary_score_cmd,
        "primary",
    )
    primary_score = read_json(OUT / "primary" / "test-score.json")
    state["primary_timing"] = primary_timing
    state["test_correct"] = primary_score["correct"]
    state["test_n"] = primary_score["n"]
    state["test_accuracy_percent"] = primary_score["accuracy"]

    log("P6 reproduction")
    state["phase"] = "P6"
    reproduction_pred = predict_command("dev/src/submission.py", "dev/runs/model_bundle", "test", "DASE7506-MP2/code/data", "dev/runs/exp_b29/reproduction/test-predictions.json")
    reproduction_score_cmd = score_command("test", "dev/runs/exp_b29/reproduction/test-predictions.json", "dev/runs/exp_b29/reproduction/test-score.json")
    reproduction_timing = timed_official_pair(
        "test",
        OUT / "reproduction" / "test-predictions.json",
        OUT / "reproduction" / "test-score.json",
        reproduction_pred,
        reproduction_score_cmd,
        "reproduction",
    )
    reproduction_score = read_json(OUT / "reproduction" / "test-score.json")
    primary_prediction = read_json(OUT / "primary" / "test-predictions.json")
    reproduction_prediction = read_json(OUT / "reproduction" / "test-predictions.json")
    same_choices = semantic_prediction(primary_prediction) == semantic_prediction(reproduction_prediction)
    same_score = all(primary_score[key] == reproduction_score[key] for key in ("protocol", "split", "correct", "n", "accuracy"))
    if not same_choices or not same_score:
        raise FreezeFailure("reproduction does not match the primary semantic result")
    if sha256_file(OUT / "freeze_manifest.json") != manifest_hash:
        raise FreezeFailure("freeze manifest changed during test")
    state["reproduction_timing"] = reproduction_timing
    state["reproducibility_verdict"] = "EXACT_MATCH"
    resource_pass = bool(primary_timing["within_limit"] and reproduction_timing["within_limit"] and assets["passed"])
    state["local_resource_verdict"] = "LOCAL_WITHIN_LIMIT" if resource_pass else "LOCAL_LIMIT_EXCEEDED"
    state["overall"] = "LOCAL_VERIFIED_REFERENCE_PENDING" if resource_pass else "FAILED"
    write_json(OUT / "resource_report.json", {
        "hardware": hardware,
        "asset_decision_bytes": assets["decision_bytes"],
        "asset_limit_bytes": ASSET_LIMIT,
        "requested_numeric_threads": 1,
        "max_compute_threads_cap": 4,
        "primary": primary_timing,
        "reproduction": reproduction_timing,
        "maximum_sum_process_wall_ns": max(primary_timing["sum_process_wall_ns"], reproduction_timing["sum_process_wall_ns"]),
        "maximum_continuous_pair_wall_ns": max(primary_timing["continuous_pair_wall_ns"], reproduction_timing["continuous_pair_wall_ns"]),
        "local_verdict": state["local_resource_verdict"],
        "reference_machine_verdict": "PENDING_NOT_RUN",
        "reference_cpu": "Xeon Platinum 8457C",
        "timing_note": "Local Windows timing only. B20 validation timing is not reused.",
    })
    write_json(OUT / "reproduction_report.json", {
        "choices_match": same_choices,
        "score_match": same_score,
        "primary_correct": primary_score["correct"],
        "primary_n": primary_score["n"],
        "primary_accuracy": primary_score["accuracy"],
        "reproduction_correct": reproduction_score["correct"],
        "reproduction_n": reproduction_score["n"],
        "reproduction_accuracy": reproduction_score["accuracy"],
        "prediction_hashes": {
            "primary": primary_timing["prediction_sha256"],
            "reproduction": reproduction_timing["prediction_sha256"],
        },
    })
    if snapshot_official() != official_snapshot:
        state["official_repository_unchanged"] = False
        raise FreezeFailure("official repository changed during test")
    state["official_repository_unchanged"] = True
    state["phase"] = "P7"
    log(f"P7 primary official accuracy {primary_score['accuracy']} ({primary_score['correct']}/{primary_score['n']})")
    return state


def main() -> None:
    state: Dict[str, Any] = {"phase": "start", "overall": "FAILED", "official_repository_unchanged": None}
    try:
        state = execute(state)
    except Exception as exc:
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        if state.get("phase") in {"P0", "P1", "P2", "start"}:
            state["overall"] = "PRESEAL_FAILED"
        elif state.get("overall") not in {"FAILED", "LOCAL_VERIFIED_REFERENCE_PENDING"}:
            state["overall"] = "FAILED"
        log(f"FAILED in {state.get('phase')}: {exc}")
        if state.get("official_snapshot") is not None:
            state["official_repository_unchanged"] = snapshot_official() == state["official_snapshot"]
        write_summary(state)
        raise
    write_summary(state)


if __name__ == "__main__":
    main()
