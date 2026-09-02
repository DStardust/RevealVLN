"""Causal evidence validation for MF3ZV progress transitions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .progress_schema import ProgressTransition


def validate_causal_evidence(
    *, decision_step: int, evidence_steps: Sequence[int], evidence_paths: Sequence[Path]
) -> None:
    if decision_step < 0:
        raise ValueError("decision_step must be non-negative")
    if not evidence_steps or len(evidence_steps) != len(evidence_paths):
        raise ValueError("evidence steps and paths must be non-empty and aligned")
    previous = -1
    for step, path in zip(evidence_steps, evidence_paths):
        if step <= previous:
            raise ValueError("causal evidence steps must be strictly increasing")
        if step > decision_step:
            raise ValueError("future evidence is forbidden")
        if not path.is_file():
            raise FileNotFoundError(path)
        previous = step


def evidence_inventory(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = path.read_bytes()
        rows.append(
            {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    return rows


def transition_from_review(row: Mapping[str, Any], root: Path) -> ProgressTransition:
    required = {
        "dataset",
        "episode_id",
        "scene_id",
        "atom_id",
        "decision_step",
        "before_step",
        "after_step",
        "state_before",
        "state_after",
        "evidence_paths",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"missing progress transition fields: {sorted(missing)}")
    paths = tuple(root / str(item) for item in row["evidence_paths"])
    validate_causal_evidence(
        decision_step=int(row["decision_step"]),
        evidence_steps=(int(row["before_step"]), int(row["after_step"])),
        evidence_paths=paths,
    )
    return ProgressTransition(
        dataset=str(row["dataset"]),
        episode_id=str(row["episode_id"]),
        scene_id=str(row["scene_id"]),
        atom_id=str(row["atom_id"]),
        before_step=int(row["before_step"]),
        after_step=int(row["after_step"]),
        state_before=str(row["state_before"]),
        state_after=str(row["state_after"]),
        evidence_paths=tuple(str(item) for item in row["evidence_paths"]),
    )

