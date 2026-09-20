"""Shared deterministic helpers for the offline errata package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


VERSION = "v16_rqf_offline_errata_v2_1"
RUN_ID = "v16_repair_rqf_002_errata_001"
HOUSE = "rqfALeAoiTq"
SPLIT = "TEST"
BASELINE_REVISION = "d84f8311b81f08f0c840308a948bdf773e0ee492"
FIXED_YAWS = (0, 3, 6, 9, 12, 15, 18, 21)
HISTORY_IDS = ("H_A", "H_B", "H_A_R", "H_B_R")
CONTINUATIONS = ("C0", "C_A", "C_B")
TRACE_IDS = tuple(f"{history}__{continuation}" for history in HISTORY_IDS for continuation in CONTINUATIONS)
TASK_IDS = ("task_A", "task_B", "task_T")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def load_json_bytes(data: bytes, identity: str = "<bytes>") -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"INVALID_JSON:{identity}:{exc}") from exc


def load_json(path: Path) -> Any:
    return load_json_bytes(Path(path).read_bytes(), str(path))


def write_new(path: Path, payload: bytes) -> None:
    """Write one new ordinary artifact and fsync it; never overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, pretty_bytes(value))


def replace_json_state(path: Path, value: Any) -> None:
    """Atomically replace only the current run's mutable pre-seal state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = pretty_bytes(value)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_within(path: Path, root: Path) -> Path:
    resolved_root = Path(root).resolve(strict=True)
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"SYMLINK_NOT_ALLOWED:{candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"PATH_OUTSIDE_ALLOWED_ROOT:{candidate}")
    return resolved

