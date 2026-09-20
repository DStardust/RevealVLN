from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from common import CONTINUATIONS, HISTORY_IDS, TASK_IDS, digest_value
from family_integrity import MappingReferenceResolver


@contextmanager
def temporary_directory(prefix: str) -> Iterator[Path]:
    parent = os.environ.get("Q35N_TEST_ROOT")
    if parent:
        Path(parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as name:
        yield Path(name)


def observation(step: int, *, pixels: dict[str, int] | None = None, suffix: str = "x") -> dict[str, Any]:
    return {
        "step": step,
        "evidence_complete": True,
        "pixels": pixels or {},
        "rgb_hash": f"rgb-{suffix}-{step}",
        "semantic_hash": f"semantic-{suffix}-{step}",
    }


def label(value: str) -> dict[str, Any]:
    return {"safe_v16_label": value, "legacy_v15_label": value}


def full_family_fixture() -> tuple[dict[str, Any], MappingReferenceResolver, dict[str, Any], dict[str, Any]]:
    values: dict[str, Any] = {}
    histories: dict[str, list[str]] = {}
    history_refs: dict[str, dict[str, str]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for history_id in HISTORY_IDS:
        own_instance = 1 if history_id.startswith("H_A") else 2
        history_observations = [
            observation(0, pixels={str(own_instance): 300}, suffix=history_id),
            observation(1, pixels={str(own_instance): 300, "3": 300}, suffix=history_id),
            observation(2, pixels={"1": 300, "2": 300, "3": 300}, suffix=history_id),
        ]
        discovery = {
            "actions": ["L", "R"],
            "observations": history_observations,
            "collisions": 0,
            "complete": True,
            "interior_state_assignments": 0,
        }
        history_path = f"fixture/{history_id}_DISCOVERY.json"
        values[history_path] = discovery
        histories[history_id] = ["L", "R"]
        history_refs[history_id] = {"path": history_path, "sha256": digest_value(discovery)}
        own = "task_A" if history_id.startswith("H_A") else "task_B"
        other = "task_B" if own == "task_A" else "task_A"
        for continuation in CONTINUATIONS:
            trace_id = f"{history_id}__{continuation}"
            observations = [dict(value) for value in history_observations]
            actions = ["L", "R", "S"]
            if continuation != "C0":
                instance = "1" if continuation == "C_A" else "2"
                observations.append(
                    observation(3, pixels={instance: 300, "3": 300}, suffix=f"{history_id}-{continuation}")
                )
                observations.append(
                    observation(4, pixels={"3": 300}, suffix=f"{history_id}-{continuation}-terminal")
                )
                actions = ["L", "R", "L", "R", "S"]
            trace = {
                "actions": actions,
                "observations": observations,
                "collisions": 0,
                "complete": True,
                "interior_state_assignments": 0,
                "cutoff": 2,
            }
            path = f"fixture/{trace_id}.json"
            values[path] = trace
            labels = {task: label("FAIL") for task in TASK_IDS}
            labels["task_T"] = label("PASS")
            labels[own] = label("PASS")
            if continuation == "C0":
                labels[other] = label("FAIL")
            elif continuation == "C_A":
                labels["task_A"] = label("PASS")
            else:
                labels["task_B"] = label("PASS")
            traces[trace_id] = {"path": path, "sha256": digest_value(trace), "labels": labels}
    proposal = {"id": "fixture-family", "start": [1.0, 2.0, 3.0], "role_indices": [0, 1, 2]}
    family = {
        "family_id": "fixture-family",
        "house": "rqfALeAoiTq",
        "split": "TEST",
        "scene": "fixture.glb",
        "roles": {"anchor_A": {}, "anchor_B": {}, "terminal": {}},
        "compiler": {
            "roles": {"anchor_A": ["chair", "room"], "anchor_B": ["bed", "room"], "terminal": ["plant", "room"]},
            "tasks": {
                "task_A": {"anchor": "anchor_A", "terminal": "terminal", "instruction": "A"},
                "task_B": {"anchor": "anchor_B", "terminal": "terminal", "instruction": "B"},
            },
            "eligible": {"anchor_A": [1], "anchor_B": [2], "terminal": [3]},
        },
        "histories": histories,
        "history_evidence_refs": history_refs,
        "traces": traces,
        "recovery": {},
        "content_root": "fixture/content",
        "proposal": proposal,
        "initial_position": proposal["start"],
        "initial_yaw": 0,
        "training_admission": "PENDING_SPLIT_AUDIT",
    }
    protocol = {"fixed_yaws": [0, 3, 6, 9, 12, 15, 18, 21]}
    provenance = {
        "house": "rqfALeAoiTq",
        "split": "TEST",
        "publication_authorized": True,
        "split_audit_status": "PENDING",
    }
    return family, MappingReferenceResolver(values), protocol, provenance


class FakeBackend:
    def __init__(self, collision_at: int | None = None):
        self.step_calls = 0
        self.observe_calls = 0
        self.reset_calls = 0
        self.collision_at = collision_at

    def reset(self, position, yaw, state):
        self.reset_calls += 1

    def observe(self):
        self.observe_calls += 1
        return {"evidence_complete": True, "pixels": {}, "rgb_hash": "r", "semantic_hash": "s"}

    def step(self, action):
        self.step_calls += 1
        return self.collision_at == self.step_calls
