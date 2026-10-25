"""Separate diagnostic probes. Events here are not method network calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys


def probe_reference(path: Path) -> None:
    try:
        path.read_bytes()
    except PermissionError as exc:
        if "reference" not in str(exc):
            raise SystemExit(f"unexpected denial: {exc}") from exc
        print(json.dumps({"probe": "synthetic_reference", "denied": True}))
        return
    raise SystemExit("synthetic reference read was not denied")


def probe_path(path: Path) -> None:
    try:
        path.read_bytes()
    except PermissionError:
        print(json.dumps({"probe": "path", "denied": True, "method_event": False}))
        return
    raise SystemExit("path read was not denied")


def probe_network() -> None:
    denied = []
    try:
        socket.getaddrinfo("example.com", 80)
    except PermissionError:
        denied.append("getaddrinfo")
    else:
        raise SystemExit("getaddrinfo was not denied")
    try:
        sock = socket.socket()
        sock.connect(("127.0.0.1", 9))
    except PermissionError:
        denied.append("connect")
    else:
        raise SystemExit("socket connect was not denied")
    print(json.dumps({"probe": "network", "denied": denied, "method_event": False}))


def probe_closure(checkpoint: Path, output: Path) -> None:
    src = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src))
    import submission  # type: ignore
    import numpy as np
    from threadpoolctl import threadpool_info

    import torch

    selected = submission.predict([], {"embeddings": np.zeros((0, 4, 1024), np.float32)}, checkpoint)
    if selected != []:
        raise SystemExit("empty bank did not return an empty choice list")
    modules = []
    for name, module in sorted(sys.modules.items()):
        file_name = getattr(module, "__file__", None)
        if not file_name:
            continue
        modules.append({"name": name, "file": str(Path(file_name).resolve())})
    output.write_text(json.dumps({
        "empty_selected": selected,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "threadpools": threadpool_info(),
        "modules": modules,
    }) + "\n", encoding="utf-8")
    print(json.dumps({"probe": "closure", "modules": len(modules), "cuda_available": bool(torch.cuda.is_available())}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("reference", "network", "closure", "path"))
    parser.add_argument("--path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.kind == "reference":
        probe_reference(args.path)
    elif args.kind == "path":
        probe_path(args.path)
    elif args.kind == "network":
        probe_network()
    else:
        probe_closure(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
