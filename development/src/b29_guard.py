"""Startup file and network guard for sealed B29 prediction processes."""
from __future__ import annotations

import os
from pathlib import Path
import sys

_INSTALLED = False
_IN_HOOK = False
_SPLITS = ("train", "validation", "test", "audit")


def _norm(path: Path) -> str:
    return os.path.normcase(str(path))


def _resolved(raw) -> Path:
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def _is_relative_to(path: Path, root: Path) -> bool:
    child = os.path.normcase(os.path.abspath(str(path)))
    parent = os.path.normcase(os.path.abspath(str(root)))
    if child == parent:
        return True
    prefix = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(prefix)


def _split_of(path: Path):
    name = path.name
    for split in _SPLITS:
        if name in {f"gsm8k_{split}.jsonl", f"gsm8k_{split}.npz"}:
            return split
    return None


def _is_write(mode) -> bool:
    if isinstance(mode, int):
        flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND
        return bool(mode & flags)
    text = str(mode).lower()
    return any(flag in text for flag in ("w", "a", "x", "+"))


def _log(message: str) -> None:
    target = os.environ.get("B29_DENY_LOG", "")
    if not target:
        return
    try:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        return


def _deny(reason: str) -> None:
    _log(reason)
    raise PermissionError(reason)


def _allowed_roots() -> list[Path]:
    roots = []
    for key in ("B29_ALLOW_ROOTS",):
        for item in os.environ.get(key, "").split(";"):
            if item:
                roots.append(_resolved(item))
    return roots


def _outside_workspace_allowed(path: Path, workspace: Path) -> bool:
    return not _is_relative_to(path, workspace)


def _check_open(raw, mode) -> None:
    if raw is None or isinstance(raw, int):
        return
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    if not text or text.startswith("<"):
        return
    path = _resolved(text)
    workspace = _resolved(os.environ["B29_WORKSPACE"])
    official = _resolved(os.environ["B29_OFFICIAL_ROOT"])
    allowed_split = os.environ.get("B29_ALLOWED_SPLIT", "")
    output = os.environ.get("B29_OUTPUT_PATH", "")
    output_path = _resolved(output) if output else None

    split = _split_of(path)
    if split is not None and path.parent.name != "candidates":
        _deny("B29 guard denied reference read")
    if split is not None and path.parent.name == "candidates" and split != allowed_split:
        _deny(f"B29 guard denied other split input: {split}")

    if _is_relative_to(path, workspace):
        name = path.name.lower()
        cached = ("prediction", "logit", "oof", "-score.json", "score.json")
        if path.suffix.lower() in {".json", ".npz"} and any(token in name for token in cached):
            if output_path is None or _norm(path) != _norm(output_path):
                _deny("B29 guard denied cached prediction read")

    if _is_relative_to(path, official) and _is_write(mode):
        _deny("B29 guard denied write to official repository")

    if _outside_workspace_allowed(path, workspace):
        return
    for root in _allowed_roots():
        if _is_relative_to(path, root):
            return
    _deny(f"B29 guard denied path outside allowlist: {path}")


def _hook(event, args):
    global _IN_HOOK
    if _IN_HOOK:
        return
    _IN_HOOK = True
    try:
        if event == "open":
            raw = args[0] if args else None
            mode = args[1] if len(args) > 1 else "r"
            _check_open(raw, mode)
        elif event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "urllib.Request",
            "http.client.connect",
            "http.client.send",
        }:
            _deny(f"B29 guard denied network event: {event}")
        elif event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
            _deny(f"B29 guard denied child process: {event}")
    finally:
        _IN_HOOK = False


def install_from_env() -> None:
    global _INSTALLED
    if _INSTALLED or os.environ.get("B29_GUARD") != "1":
        return
    required = ("B29_WORKSPACE", "B29_OFFICIAL_ROOT", "B29_ALLOWED_SPLIT", "B29_ALLOW_ROOTS")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"B29 guard missing environment: {missing}")
    sys.addaudithook(_hook)
    _INSTALLED = True
    print("B29_GUARD_ACTIVE", file=sys.stderr, flush=True)
