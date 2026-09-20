"""Actual source/dependency hashing for the isolated offline entrypoint."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def file_identity(path: Path, worktree: Path) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    try:
        relative = str(path.relative_to(worktree.resolve(strict=True)))
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "resolved_path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_source_lock(
    files: Iterable[Path],
    worktree: Path,
    baseline_commit: str,
    runtime_modules: dict[str, str],
) -> dict[str, object]:
    identities = [file_identity(path, worktree) for path in files]
    identities.sort(key=lambda row: str(row["path"]))
    return {
        "version": "v16_rqf_offline_errata_v2_1",
        "baseline_commit": baseline_commit,
        "entrypoint": str((Path(worktree) / "projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/rqf_offline_errata_v2_1/driver.py").resolve()),
        "files": identities,
        "runtime_module_files": runtime_modules,
        "generated_outputs_excluded_to_avoid_self_reference": True,
    }

