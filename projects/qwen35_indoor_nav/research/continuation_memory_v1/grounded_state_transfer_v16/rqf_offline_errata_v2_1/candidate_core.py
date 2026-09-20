"""Future shared candidate finalization and CPU-testable bounded control flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import pretty_bytes
from family_integrity import ReferenceResolver, validate_new_family_for_commit
from immutable_publish import publish_bytes_no_replace
from snapshot_io import EvidenceInvalid, EvidenceMissing


class PhysicalRejected(RuntimeError):
    """A confirmed physical gate rejection, shared by future entrypoints."""


def classify_exception(exc: BaseException, *, boundary: str) -> str:
    if isinstance(exc, PhysicalRejected):
        return "PHYSICAL_REJECTED"
    if isinstance(exc, EvidenceMissing):
        return "EVIDENCE_MISSING"
    if isinstance(exc, (EvidenceInvalid, json.JSONDecodeError)):
        return "EVIDENCE_INVALID"
    if isinstance(exc, FileNotFoundError) and boundary == "frozen_input":
        return "EVIDENCE_MISSING"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "INTERRUPTED"
    return "ERROR"


def finalize_candidate_result(
    candidate_result: dict[str, Any],
    resolver: ReferenceResolver,
    output_root: Path,
    provenance: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete family, serialize once, and publish exactly once."""
    family = candidate_result.get("family")
    validation = validate_new_family_for_commit(family, resolver, protocol, provenance)
    if not validation["publication_eligible"]:
        raise PermissionError("PUBLICATION_NOT_AUTHORIZED")
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("OUTPUT_ROOT")
    destination = output_root / family["family_id"] / "FAMILY.json"
    destination.parent.mkdir(parents=False, exist_ok=True)
    payload = pretty_bytes(family)
    publish_bytes_no_replace(destination, payload)
    return {
        "status": "PUBLISHED",
        "destination": str(destination),
        "publication_calls": 1,
        "validation": validation,
    }


class BoundedRunner:
    """The action-count and collision boundary used by CPU behavior tests."""

    def __init__(self, backend: Any, cap: int):
        if type(cap) is not int or cap <= 0:
            raise ValueError("RUNNER_CAP")
        self.backend = backend
        self.cap = cap
        self.trace: dict[str, Any] = {}

    def start(self, position: list[float], yaw: int = 0) -> None:
        self.backend.reset(position, yaw, 0)
        self.trace = {
            "actions": [],
            "observations": [],
            "collisions": 0,
            "complete": False,
            "interior_state_assignments": 0,
            "initial_position": position,
            "initial_yaw": yaw,
        }
        self._observe()

    def _observe(self) -> dict[str, Any]:
        observation = dict(self.backend.observe(), step=len(self.trace["observations"]))
        self.trace["observations"].append(observation)
        if observation.get("evidence_complete") is not True:
            raise PhysicalRejected("UNKNOWN_SEMANTIC_MASK")
        return observation

    def move(self, action: str) -> None:
        if action not in {"F", "L", "R", "S"}:
            raise ValueError("ACTION")
        if len(self.trace.get("actions", [])) >= self.cap:
            raise PhysicalRejected("DECISION_BUDGET")
        if "S" in self.trace["actions"]:
            raise PhysicalRejected("ACTION_AFTER_STOP")
        self.trace["actions"].append(action)
        if action == "S":
            return
        collision = bool(self.backend.step(action))
        self.trace["collisions"] += int(collision)
        self._observe()
        if collision:
            raise PhysicalRejected("COLLISION")

    def finish(self, *, terminal_seen_at_stop: bool, event_order_satisfied: bool) -> dict[str, Any]:
        actions = self.trace.get("actions", [])
        if actions[-1:] != ["S"]:
            raise PhysicalRejected("ACTIVE_STOP_REQUIRED")
        if len(actions) > self.cap:
            raise PhysicalRejected("DECISION_BUDGET")
        if not terminal_seen_at_stop or not event_order_satisfied:
            raise PhysicalRejected("STOP_TERMINATION_CONDITION")
        self.trace["complete"] = True
        return self.trace


def validate_history_isolation(*, own_anchor_seen: bool, other_anchor_seen: bool) -> None:
    if not own_anchor_seen or other_anchor_seen:
        raise PhysicalRejected("HISTORY_EVENT_STATE_NOT_ISOLATED")


def preserve_diagnostic_failure(primary: BaseException, recorder: Any) -> dict[str, Any]:
    """Preserve both errors if recording the primary failure also fails."""
    try:
        recorder(primary)
    except BaseException as secondary:
        return {
            "status": "ERROR",
            "primary_error": repr(primary),
            "diagnostic_recording_error": repr(secondary),
        }
    return {"status": classify_exception(primary, boundary="runtime"), "primary_error": repr(primary)}


def run_registered_operations(operations: list[Any]) -> dict[str, Any]:
    """Run only the pre-registered operations; interruption stops immediately."""
    completed = 0
    for operation in operations:
        try:
            operation()
        except (KeyboardInterrupt, SystemExit) as exc:
            return {"status": "INTERRUPTED", "completed": completed, "error": repr(exc)}
        except PhysicalRejected as exc:
            completed += 1
            continue
        except BaseException as exc:
            return {"status": "ERROR", "completed": completed, "error": repr(exc)}
        completed += 1
    return {"status": "COMPLETE", "completed": completed}
