"""Strict future-family validation and read-only existing-publication inspection."""
from __future__ import annotations

import json
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from common import (
    CONTINUATIONS,
    HISTORY_IDS,
    TASK_IDS,
    TRACE_IDS,
    canonical_bytes,
    digest_bytes,
    digest_value,
)
from snapshot_io import EvidenceInvalid, EvidenceMissing, SnapshotReader


class FamilyValidationError(ValueError):
    pass


class ReferenceResolver(Protocol):
    def read_json_ref(self, reference: dict[str, Any]) -> Any: ...


class SnapshotReferenceResolver:
    def __init__(self, reader: SnapshotReader):
        self.reader = reader

    def read_json_ref(self, reference: dict[str, Any]) -> Any:
        if not isinstance(reference, dict) or not {"path", "sha256"}.issubset(reference):
            raise FamilyValidationError("REFERENCE_SCHEMA")
        path, expected = reference["path"], reference["sha256"]
        if not isinstance(path, str) or not isinstance(expected, str):
            raise FamilyValidationError("REFERENCE_TYPES")
        record = self.reader.record_for_path(path)
        if record["object_sha256"] != expected:
            raise FamilyValidationError(f"REFERENCE_HASH:{path}")
        return self.reader.read_json(expected)


class MappingReferenceResolver:
    """In-memory CPU fixture resolver using the same path+hash contract."""

    def __init__(self, values: dict[str, Any]):
        self.values = values

    def read_json_ref(self, reference: dict[str, Any]) -> Any:
        if not isinstance(reference, dict) or not {"path", "sha256"}.issubset(reference):
            raise FamilyValidationError("REFERENCE_SCHEMA")
        path = reference["path"]
        if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
            raise FamilyValidationError("REFERENCE_PATH")
        if path not in self.values:
            raise FileNotFoundError(path)
        value = self.values[path]
        if digest_value(value) != reference["sha256"]:
            raise FamilyValidationError(f"REFERENCE_HASH:{path}")
        return value


@lru_cache(maxsize=1)
def _frozen_validation_implementation() -> tuple[type, Any]:
    v16 = Path(__file__).resolve().parent.parent
    project = v16.parents[2]
    compiler_path = project / "data_pipeline/mechanism_factory_v2/compiler.py"
    compiler_spec = importlib.util.spec_from_file_location("q35n_errata_family_compiler", compiler_path)
    if compiler_spec is None or compiler_spec.loader is None:
        raise RuntimeError(f"COMPILER_IMPORT:{compiler_path}")
    compiler_module = importlib.util.module_from_spec(compiler_spec)
    compiler_spec.loader.exec_module(compiler_module)
    if str(v16) not in sys.path:
        sys.path.insert(0, str(v16))
    evaluator_path = v16 / "evaluator_v16.py"
    evaluator_spec = importlib.util.spec_from_file_location("q35n_errata_family_evaluator", evaluator_path)
    if evaluator_spec is None or evaluator_spec.loader is None:
        raise RuntimeError(f"EVALUATOR_IMPORT:{evaluator_path}")
    evaluator_module = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator_module)
    return compiler_module.Compiler, evaluator_module.evaluate


def _fail(code: str, detail: object = "") -> None:
    suffix = f":{detail}" if detail != "" else ""
    raise FamilyValidationError(code + suffix)


def _actions(value: Any, *, cap: int, stop_required: bool) -> list[str]:
    if not isinstance(value, list) or any(action not in {"F", "L", "R", "S"} for action in value):
        _fail("TRACE_ACTIONS")
    if len(value) > cap:
        _fail("TRACE_BUDGET", len(value))
    if "S" in value[:-1]:
        _fail("STOP_BEFORE_END")
    if stop_required and value[-1:] != ["S"]:
        _fail("ACTIVE_STOP_REQUIRED")
    if not stop_required and "S" in value:
        _fail("DISCOVERY_STOP_FORBIDDEN")
    return value


def _observations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("TRACE_OBSERVATIONS")
    for index, observation in enumerate(value):
        if not isinstance(observation, dict) or observation.get("step") != index:
            _fail("OBSERVATION_INDEX", index)
        if observation.get("evidence_complete") is not True:
            _fail("OBSERVATION_EVIDENCE", index)
        if not isinstance(observation.get("pixels"), dict):
            _fail("OBSERVATION_PIXELS", index)
    return value


def _observation_identity(observation: dict[str, Any]) -> tuple[Any, Any]:
    rgb, semantic = observation.get("rgb_hash"), observation.get("semantic_hash")
    if not isinstance(rgb, str) or not isinstance(semantic, str):
        _fail("OBSERVATION_CONTENT_IDENTITY")
    return rgb, semantic


def _validate_trace_common(trace: Any, *, cap: int, stop_required: bool) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(trace, dict) or trace.get("complete") is not True:
        _fail("TRACE_INCOMPLETE")
    collisions = trace.get("collisions")
    if type(collisions) is not int or collisions != 0:
        _fail("TRACE_COLLISION", collisions)
    if trace.get("interior_state_assignments", 0) != 0:
        _fail("INTERIOR_STATE_ASSIGNMENT")
    actions = _actions(trace.get("actions"), cap=cap, stop_required=stop_required)
    observations = _observations(trace.get("observations"))
    expected_observations = len(actions) if stop_required else len(actions) + 1
    if len(observations) != expected_observations:
        _fail("ACTION_OBSERVATION_LENGTH", f"{len(actions)}:{len(observations)}")
    return actions, observations


def _validate_labels(trace_id: str, labels: Any) -> None:
    if not isinstance(labels, dict) or set(labels) != set(TASK_IDS):
        _fail("LABEL_TASK_KEYS", trace_id)
    for task_id, label in labels.items():
        if not isinstance(label, dict):
            _fail("LABEL_SCHEMA", f"{trace_id}:{task_id}")
        if label.get("safe_v16_label") not in {"PASS", "FAIL"}:
            _fail("LABEL_UNKNOWN", f"{trace_id}:{task_id}")
        if label.get("legacy_v15_label") not in {"PASS", "FAIL"}:
            _fail("LEGACY_LABEL_UNKNOWN", f"{trace_id}:{task_id}")


def validate_existing_family_reference(
    family: Any, resolver: ReferenceResolver | None, source_identity: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(family, dict) or not isinstance(family.get("family_id"), str):
        _fail("EXISTING_FAMILY_SCHEMA")
    actual = digest_value(family)
    expected = source_identity.get("canonical_sha256")
    if expected is not None and actual != expected:
        _fail("EXISTING_FAMILY_VALUE_IDENTITY")
    references = family.get("traces")
    checked, missing = 0, 0
    if resolver is not None and isinstance(references, dict):
        for trace in references.values():
            if not isinstance(trace, dict) or "path" not in trace or "sha256" not in trace:
                missing += 1
                continue
            try:
                resolver.read_json_ref({"path": trace["path"], "sha256": trace["sha256"]})
                checked += 1
            except (FileNotFoundError, FamilyValidationError, EvidenceMissing, EvidenceInvalid):
                missing += 1
    return {
        "family_id": family["family_id"],
        "canonical_sha256": actual,
        "value_equal": expected is None or actual == expected,
        "references_checked": checked,
        "references_unavailable_or_invalid": missing,
        "recollected": False,
        "newly_certified": False,
    }


def validate_new_family_for_commit(
    family: Any,
    resolver: ReferenceResolver,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(family, dict):
        _fail("FAMILY_NOT_OBJECT")
    required = {
        "family_id",
        "house",
        "split",
        "scene",
        "roles",
        "compiler",
        "histories",
        "history_evidence_refs",
        "traces",
        "recovery",
        "content_root",
        "proposal",
        "initial_position",
        "initial_yaw",
        "training_admission",
    }
    missing = sorted(required - set(family))
    if missing:
        _fail("FAMILY_INCOMPLETE", ",".join(missing))
    if not isinstance(family["family_id"], str) or not family["family_id"]:
        _fail("FAMILY_ID")
    if family["house"] != provenance.get("house") or family["split"] != provenance.get("split"):
        _fail("FAMILY_SOURCE_IDENTITY")
    proposal = family["proposal"]
    if not isinstance(proposal, dict) or proposal.get("id") != family["family_id"]:
        _fail("FAMILY_PROPOSAL_IDENTITY")
    if family["initial_position"] != proposal.get("start") or family["initial_yaw"] not in protocol["fixed_yaws"]:
        _fail("FAMILY_INITIAL_STATE")
    if not isinstance(family["content_root"], str) or not family["content_root"]:
        _fail("FAMILY_CONTENT_ROOT")
    if not isinstance(family["roles"], dict) or set(family["roles"]) != {"anchor_A", "anchor_B", "terminal"}:
        _fail("FAMILY_ROLES")
    compiler = family["compiler"]
    if not isinstance(compiler, dict) or set(compiler.get("tasks", {})) != {"task_A", "task_B"}:
        _fail("FAMILY_COMPILER_TASKS")
    if set(compiler.get("roles", {})) != set(family["roles"]) or set(compiler.get("eligible", {})) != set(family["roles"]):
        _fail("FAMILY_COMPILER_ROLES")

    histories = family["histories"]
    history_refs = family["history_evidence_refs"]
    traces = family["traces"]
    if not isinstance(histories, dict) or set(histories) != set(HISTORY_IDS):
        _fail("HISTORY_KEY_SET")
    if not isinstance(history_refs, dict) or set(history_refs) != set(HISTORY_IDS):
        _fail("HISTORY_REFERENCE_KEY_SET")
    if not isinstance(traces, dict) or set(traces) != set(TRACE_IDS):
        _fail("TRACE_KEY_SET")

    discovery: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    for history_id in HISTORY_IDS:
        reference = history_refs[history_id]
        value = resolver.read_json_ref(reference)
        actions, observations = _validate_trace_common(value, cap=240, stop_required=False)
        if actions != histories[history_id]:
            _fail("HISTORY_ACTION_REFERENCE_MISMATCH", history_id)
        discovery[history_id] = actions, observations

    loaded_traces: dict[str, dict[str, Any]] = {}
    loaded_trace_values: dict[str, dict[str, Any]] = {}
    for trace_id in TRACE_IDS:
        trace_record = traces[trace_id]
        if not isinstance(trace_record, dict) or not {"path", "sha256", "labels"}.issubset(trace_record):
            _fail("TRACE_REFERENCE_SCHEMA", trace_id)
        value = resolver.read_json_ref({"path": trace_record["path"], "sha256": trace_record["sha256"]})
        actions, observations = _validate_trace_common(value, cap=500, stop_required=True)
        history_id = trace_id.split("__", 1)[0]
        history_actions, history_observations = discovery[history_id]
        cutoff = value.get("cutoff")
        if cutoff != len(history_actions):
            _fail("TRACE_CUTOFF", trace_id)
        if actions[:cutoff] != history_actions:
            _fail("TRACE_HISTORY_ACTION_PREFIX", trace_id)
        expected_prefix_length = len(history_observations)
        if len(observations) < expected_prefix_length:
            _fail("TRACE_HISTORY_OBSERVATION_PREFIX_LENGTH", trace_id)
        for index in range(expected_prefix_length):
            if _observation_identity(observations[index]) != _observation_identity(history_observations[index]):
                _fail("TRACE_HISTORY_OBSERVATION_PREFIX", f"{trace_id}:{index}")
        _validate_labels(trace_id, trace_record["labels"])
        loaded_traces[trace_id] = trace_record
        loaded_trace_values[trace_id] = value

    for history_id in HISTORY_IDS:
        own = "task_A" if history_id.startswith("H_A") else "task_B"
        other = "task_B" if own == "task_A" else "task_A"
        c0 = loaded_traces[f"{history_id}__C0"]["labels"]
        if c0[own]["safe_v16_label"] != "PASS" or c0[other]["safe_v16_label"] != "FAIL":
            _fail("C0_HISTORY_SENSITIVE_LABELS", history_id)
        for task_id in TASK_IDS:
            if not any(
                loaded_traces[f"{history_id}__{continuation}"]["labels"][task_id]["safe_v16_label"] == "PASS"
                for continuation in CONTINUATIONS
            ):
                _fail("NO_PASS_TEACHER", f"{history_id}:{task_id}")

    compiler_class, frozen_evaluate = _frozen_validation_implementation()
    try:
        frozen_compiler = compiler_class(**compiler)
    except (TypeError, ValueError) as exc:
        _fail("FROZEN_COMPILER_CONFIG", repr(exc))
    for trace_id, trace_record in loaded_traces.items():
        value = loaded_trace_values[trace_id]
        cutoff = value["cutoff"]
        for task_id in TASK_IDS:
            recomputed = frozen_evaluate(frozen_compiler, value, task_id, cutoff)
            declared = trace_record["labels"][task_id]
            for key in ("legacy_v15_label", "safe_v16_label"):
                if declared.get(key) != recomputed.get(key):
                    _fail("LABEL_RECOMPUTE_MISMATCH", f"{trace_id}:{task_id}:{key}")

    return {
        "schema_valid": True,
        "references_verified": True,
        "physical_checks_verified": True,
        "label_matrix_verified": True,
        "labels_recomputed_with_frozen_evaluator": True,
        "split_audit_status": provenance.get("split_audit_status", "PENDING"),
        "training_admission_status": family["training_admission"],
        "publication_eligible": provenance.get("publication_authorized") is True,
        "newly_admitted_in_this_run": False,
        "family_canonical_sha256": digest_value(family),
    }


def inspect_existing_publication(
    path: Path,
    expected_family: dict[str, Any],
    resolver: ReferenceResolver,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"status": "FINAL_FILE_ABSENT"}
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED_PARTIAL_PUBLICATION", "error": repr(exc)}
    if digest_bytes(canonical_bytes(value)) != digest_bytes(canonical_bytes(expected_family)):
        return {"status": "BLOCKED_PUBLICATION_CONFLICT"}
    try:
        validation = validate_new_family_for_commit(value, resolver, protocol, provenance)
    except (FamilyValidationError, FileNotFoundError) as exc:
        return {"status": "BLOCKED_PUBLICATION_INTEGRITY", "error": repr(exc)}
    return {"status": "RESUMED_VERIFIED", "validation": validation}
