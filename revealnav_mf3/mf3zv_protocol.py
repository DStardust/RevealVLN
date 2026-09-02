"""Protocol helpers and fixed gates for MF3ZV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REVISION = "mf3zv_minimal_progress_support_v1"
STATUS_SEALED = "SEALED_BEFORE_MF3ZV_SUPPORT_RESULTS"
PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}
GATES = {
    "atom_coverage_minimum": 0.70,
    "minimum_valid_atoms": 50,
    "state_coverage_minimum": 0.70,
    "minimum_state_supported_episodes": 40,
    "minimum_state_scenes": 15,
    "minimum_local_targets_per_domain": 30,
    "minimum_local_target_scenes_per_domain": 10,
}


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("revision") != REVISION:
        raise ValueError("wrong MF3ZV revision")
    if protocol.get("status") != STATUS_SEALED:
        raise ValueError("MF3ZV protocol is not pre-result sealed")
    if protocol.get("public_split_access") != PUBLIC_CLOSED:
        raise ValueError("public splits must be fail-closed")
    if protocol.get("gates") != GATES:
        raise ValueError("MF3ZV gates differ from the sealed constants")
    for key in ("training_run", "navigation_run", "checkpoint_generated"):
        if protocol.get(key) is not False:
            raise ValueError(f"{key} must be false")
    if protocol.get("review", {}).get("maximum") != 100:
        raise ValueError("review maximum must be 100")
    if protocol.get("progress_families") != ["ORDINAL", "PASSED_LANDMARK"]:
        raise ValueError("only the two frozen MF3ZV families are allowed")


def atom_gate(valid: int, reviewed: int) -> bool:
    return reviewed > 0 and valid >= GATES["minimum_valid_atoms"] and valid / reviewed >= GATES[
        "atom_coverage_minimum"
    ]


def state_gate(supported: int, valid: int, scenes: int) -> bool:
    return (
        valid > 0
        and supported >= GATES["minimum_state_supported_episodes"]
        and supported / valid >= GATES["state_coverage_minimum"]
        and scenes >= GATES["minimum_state_scenes"]
    )


def eligible_domains(target_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    eligible = []
    for dataset in ("R2R", "RxR"):
        rows = [row for row in target_rows if row["dataset"] == dataset]
        if (
            len(rows) >= GATES["minimum_local_targets_per_domain"]
            and len({row["scene_id"] for row in rows})
            >= GATES["minimum_local_target_scenes_per_domain"]
        ):
            eligible.append(dataset)
    return eligible


def final_status(domains: Sequence[str]) -> str:
    values = set(domains)
    if values == {"R2R", "RxR"}:
        return "MF3ZV_PROGRESS_SUPPORT_PASS_BOTH"
    if values == {"R2R"}:
        return "MF3ZV_PROGRESS_SUPPORT_PASS_R2R_ONLY"
    if values == {"RxR"}:
        return "MF3ZV_PROGRESS_SUPPORT_PASS_RXR_ONLY"
    return "MF3ZV_PROGRESS_SUPPORT_FAIL"

