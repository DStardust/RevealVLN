"""Correct neutral-start and saved-history semantics over frozen objects."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from common import FIXED_YAWS
from snapshot_io import EvidenceInvalid, EvidenceMissing, SnapshotReader


REQUIRED_EVENT_ROLES = {"anchor_A", "anchor_B", "terminal"}


def classify_neutral_events(events: object) -> dict[str, object]:
    """Classify a completed L,R probe; terminal is diagnostic, not a rejector."""
    if not isinstance(events, (list, tuple)) or not events:
        raise ValueError("EMPTY_OR_INVALID_COMPILER_EVENTS")
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict) or not REQUIRED_EVENT_ROLES.issubset(event):
            raise ValueError("INVALID_COMPILER_EVENT_SCHEMA")
        for role in REQUIRED_EVENT_ROLES:
            ids = event[role]
            if not isinstance(ids, (list, tuple)):
                raise ValueError("INVALID_COMPILER_EVENT_IDS")
            if any(type(instance) is not int or instance < 0 for instance in ids):
                raise ValueError("INVALID_COMPILER_INSTANCE_ID")
            if ids:
                seen.add(role)
    anchors = sorted(seen.intersection({"anchor_A", "anchor_B"}))
    return {
        "classification": "INVALID_ANCHOR_PRESENT" if anchors else "VALID_NEUTRAL_START",
        "anchor_roles_triggered": anchors,
        "all_roles_triggered": sorted(seen),
        "terminal_seen": "terminal" in seen,
    }


def load_compiler_class(path: Path) -> type:
    spec = importlib.util.spec_from_file_location("q35n_errata_fixed_compiler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"COMPILER_IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Compiler


def _tasks(roles: dict[str, dict[str, Any]], split: str) -> dict[str, dict[str, str]]:
    terminal = roles["terminal"]
    terminal_name = f"the {terminal['raw_match']['value']} in the {terminal['room']}"
    tasks: dict[str, dict[str, str]] = {}
    for task_id, role in (("task_A", "anchor_A"), ("task_B", "anchor_B")):
        anchor = roles[role]
        anchor_name = f"the {anchor['raw_match']['value']} in the {anchor['room']}"
        if split == "FIT":
            text = f"First see {anchor_name} in two consecutive observations, then see {terminal_name} in two consecutive observations, and stop immediately."
        elif split == "DEV":
            text = f"Before stopping at {terminal_name}, establish a two-observation sighting of {anchor_name}. Stop only on a subsequent two-observation sighting of {terminal_name}."
        else:
            text = f"Complete these sightings in order: {anchor_name}, followed by {terminal_name}. Each sighting requires two consecutive views. End by stopping on the final sighting."
        tasks[task_id] = {"anchor": role, "terminal": "terminal", "instruction": text}
    return tasks


def make_compiler(proposal: dict[str, Any], house: dict[str, Any], compiler_class: type) -> Any:
    indices = proposal.get("role_indices")
    if not isinstance(indices, list) or len(indices) != 3:
        raise ValueError("PROPOSAL_ROLE_INDICES")
    roles: dict[str, dict[str, Any]] = {}
    eligible: dict[str, list[int]] = {}
    for role, index in zip(("anchor_A", "anchor_B", "terminal"), indices):
        if type(index) is not int or not 0 <= index < len(house["roles"]):
            raise ValueError("PROPOSAL_ROLE_INDEX")
        row = house["roles"][index]
        roles[role] = row["spec"]
        eligible[role] = list(row["eligible"])
    return compiler_class(
        roles={name: [value["mpcat40"], value["room"]] for name, value in roles.items()},
        tasks=_tasks(roles, house["split"]),
        eligible=eligible,
    )


def _pixel(observation: dict[str, Any], instance: int) -> int:
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError("OBSERVATION_PIXELS")
    value = pixels.get(str(instance), pixels.get(instance, 0))
    if type(value) is not int or value < 0:
        raise ValueError("OBSERVATION_PIXEL_COUNT")
    return value


def event_details(
    observations: list[dict[str, Any]], events: list[dict[str, Any]], trace_identity: str
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for current_index, event in enumerate(events):
        if current_index == 0:
            continue
        for role in sorted(REQUIRED_EVENT_ROLES):
            ids = event[role]
            for instance in ids:
                details.append(
                    {
                        "role": role,
                        "instance": instance,
                        "previous_pixels": _pixel(observations[current_index - 1], instance),
                        "current_pixels": _pixel(observations[current_index], instance),
                        "previous_observation_index": current_index - 1,
                        "current_observation_index": current_index,
                        "same_instance": True,
                        "same_trace": True,
                        "trace_identity": trace_identity,
                    }
                )
    return details


def _observation_structure(observations: Any) -> bool:
    if not isinstance(observations, list) or not observations:
        return False
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            return False
        if observation.get("step") != index or observation.get("evidence_complete") is not True:
            return False
        if not isinstance(observation.get("pixels"), dict):
            return False
    return True


def classify_probe(
    trace: Any, compiler: Any, *, trace_identity: str, evidence_status: str = "READ"
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "probe_evidence_status": evidence_status,
        "probe_gate_reached": False,
        "neutral_predicate_on_available_frames": None,
        "neutral_gate_result": "EVIDENCE_INVALID",
        "anchor_roles_triggered": [],
        "all_roles_triggered": [],
        "terminal_seen": False,
        "trigger_details": [],
    }
    if not isinstance(trace, dict):
        return result
    actions = trace.get("actions")
    observations = trace.get("observations")
    result["probe_actions"] = actions if isinstance(actions, list) else None
    if not _observation_structure(observations):
        return result
    try:
        events = compiler.atoms(observations)
        predicate = classify_neutral_events(events)
        result.update(predicate)
        result["neutral_predicate_on_available_frames"] = predicate["classification"]
        result["trigger_details"] = event_details(observations, events, trace_identity)
    except (KeyError, TypeError, ValueError) as exc:
        result["diagnostic_error"] = repr(exc)
        return result
    collisions = trace.get("collisions")
    if type(collisions) is int and collisions > 0:
        result["neutral_gate_result"] = "NOT_REACHED"
        result["probe_evidence_status"] = "PHYSICAL_REJECTED_BEFORE_GATE"
        return result
    gate_complete = (
        trace.get("complete") is True
        and collisions == 0
        and actions == ["L", "R"]
        and len(observations) == 3
    )
    if not gate_complete:
        result["neutral_gate_result"] = "NOT_REACHED"
        return result
    result["probe_gate_reached"] = True
    result["neutral_gate_result"] = result["classification"]
    return result


def aggregate_yaws(records: list[dict[str, Any]], fixed_yaws: tuple[int, ...] = FIXED_YAWS) -> dict[str, Any]:
    slots: dict[int, dict[str, Any]] = {}
    for record in records:
        yaw = record.get("yaw")
        if type(yaw) is not int or yaw not in fixed_yaws:
            raise ValueError(f"INVALID_YAW_IDENTITY:{yaw!r}")
        if yaw in slots:
            raise ValueError(f"DUPLICATE_YAW_IDENTITY:{yaw}")
        slots[yaw] = record
    unresolved: list[int] = []
    valid: list[int] = []
    invalid: list[int] = []
    for yaw in fixed_yaws:
        record = slots.get(yaw)
        gate = record.get("neutral_gate_result") if record else "NOT_REACHED"
        if gate == "VALID_NEUTRAL_START":
            valid.append(yaw)
        elif gate == "INVALID_ANCHOR_PRESENT":
            invalid.append(yaw)
        else:
            unresolved.append(yaw)
    if unresolved:
        classification = "INCOMPLETE_EVIDENCE"
    elif len(valid) == len(fixed_yaws):
        classification = "ALL_EIGHT_NEUTRAL_START_VALID"
    elif len(invalid) == len(fixed_yaws):
        classification = "ALL_EIGHT_INVALID"
    else:
        classification = "MIXED_VALID_INVALID"
    return {
        "proposal_classification": classification,
        "known_valid_yaws": valid,
        "known_invalid_yaws": invalid,
        "unresolved_yaws": unresolved,
        "expected_yaw_count": len(fixed_yaws),
        "present_record_count": len(records),
    }


def neutral_start_audit(
    reader: SnapshotReader,
    old_audit: dict[str, Any],
    proposals: list[dict[str, Any]],
    house: dict[str, Any],
    compiler_class: type,
) -> dict[str, Any]:
    proposal_by_id = {str(row["id"]): row for row in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("DUPLICATE_PROPOSAL_ID")
    old_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    for old in old_audit.get("yaw_records", []):
        key = (str(old.get("proposal")), old.get("yaw"))
        if key in old_by_slot:
            raise ValueError(f"DUPLICATE_OLD_YAW:{key}")
        old_by_slot[key] = old
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for proposal_id in sorted(proposal_by_id):
        proposal = proposal_by_id[proposal_id]
        compiler = make_compiler(proposal, house, compiler_class)
        for yaw in FIXED_YAWS:
            old = old_by_slot.get((proposal_id, yaw))
            base: dict[str, Any] = {
                "candidate_key": f"{proposal_id}:{yaw}",
                "source_run": "v16_repair_rqf_001",
                "proposal_id": proposal_id,
                "yaw": yaw,
                "attempt_id": f"{proposal_id}:yaw_{yaw:02d}:attempt_1",
                "original_outcome_raw": old.get("original_checker_result") if old else None,
                "old_classification": old.get("classification") if old else "NOT_RECORDED",
                "input_object_sha256": old.get("evidence_sha256") if old else None,
            }
            if old is None or not old.get("evidence_path"):
                base.update(
                    {
                        "probe_evidence_status": "EVIDENCE_MISSING",
                        "probe_actions": None,
                        "probe_gate_reached": False,
                        "neutral_predicate_on_available_frames": None,
                        "neutral_gate_result": "EVIDENCE_MISSING",
                        "anchor_roles_triggered": [],
                        "all_roles_triggered": [],
                        "terminal_seen": False,
                        "trigger_details": [],
                    }
                )
            else:
                expected = old.get("evidence_sha256")
                try:
                    record = reader.record_for_path(old["evidence_path"])
                    if record["object_sha256"] != expected:
                        raise EvidenceInvalid(
                            f"PROBE_EXPECTED_HASH:{old['evidence_path']}:{expected}:{record['object_sha256']}"
                        )
                    trace = reader.read_json(expected)
                    base.update(classify_probe(trace, compiler, trace_identity=expected))
                except EvidenceMissing as exc:
                    base.update(
                        {
                            "probe_evidence_status": "EVIDENCE_MISSING",
                            "probe_actions": None,
                            "probe_gate_reached": False,
                            "neutral_predicate_on_available_frames": None,
                            "neutral_gate_result": "EVIDENCE_MISSING",
                            "anchor_roles_triggered": [],
                            "all_roles_triggered": [],
                            "terminal_seen": False,
                            "trigger_details": [],
                            "diagnostic_error": repr(exc),
                        }
                    )
                except (EvidenceInvalid, ValueError) as exc:
                    base.update(
                        {
                            "probe_evidence_status": "EVIDENCE_INVALID",
                            "probe_actions": None,
                            "probe_gate_reached": False,
                            "neutral_predicate_on_available_frames": None,
                            "neutral_gate_result": "EVIDENCE_INVALID",
                            "anchor_roles_triggered": [],
                            "all_roles_triggered": [],
                            "terminal_seen": False,
                            "trigger_details": [],
                            "diagnostic_error": repr(exc),
                        }
                    )
            counts[base["neutral_gate_result"]] += 1
            rows.append(base)
    rollup: list[dict[str, Any]] = []
    for proposal_id in sorted(proposal_by_id):
        ours = [row for row in rows if row["proposal_id"] == proposal_id]
        aggregate = aggregate_yaws(ours)
        aggregate["proposal_id"] = proposal_id
        aggregate["hub"] = proposal_by_id[proposal_id].get("hub")
        rollup.append(aggregate)
    aggregate_counts = Counter(row["proposal_classification"] for row in rollup)
    return {
        "version": "v16_rqf_offline_errata_v2_1",
        "house": house["house"],
        "split": house["split"],
        "fixed_yaws": list(FIXED_YAWS),
        "yaw_records": rows,
        "yaw_gate_counts": dict(sorted(counts.items())),
        "proposal_rollup": rollup,
        "proposal_classification_counts": dict(sorted(aggregate_counts.items())),
        "terminal_is_diagnostic_not_anchor_gate": True,
        "raw_array_integrity": "NOT_RECHECKED",
    }


def _trace_event_summary(
    trace: Any, compiler: Any, history_id: str, trace_identity: str
) -> dict[str, Any]:
    if not isinstance(trace, dict) or not _observation_structure(trace.get("observations")):
        raise EvidenceInvalid("HISTORY_TRACE_STRUCTURE")
    observations = trace["observations"]
    events = compiler.atoms(observations)
    classify_neutral_events(events)
    details = event_details(observations, events, trace_identity)
    own_role = "anchor_A" if history_id.startswith("H_A") else "anchor_B"
    other_role = "anchor_B" if own_role == "anchor_A" else "anchor_A"
    own_hits = [row for row in details if row["role"] == own_role]
    other_hits = [row for row in details if row["role"] == other_role]
    actions = trace.get("actions")
    collisions = trace.get("collisions")
    evidence_complete = (
        isinstance(actions, list)
        and type(collisions) is int
        and len(observations) == len(actions) + 1
    )
    execution_complete = evidence_complete and trace.get("complete") is True
    if not own_hits:
        legacy = "OWN_ANCHOR_NOT_SEEN"
    elif other_hits:
        legacy = "RECOMPUTED_OTHER_ANCHOR_PRESENT"
    else:
        legacy = "ISOLATED_FOR_SAVED_TRACE"
    return {
        "own_anchor_seen": bool(own_hits),
        "other_anchor_seen": bool(other_hits),
        "first_own_anchor_step": own_hits[0]["current_observation_index"] if own_hits else None,
        "first_other_anchor_step": other_hits[0]["current_observation_index"] if other_hits else None,
        "own_anchor_first_trigger": own_hits[0] if own_hits else None,
        "other_anchor_first_trigger": other_hits[0] if other_hits else None,
        "legacy_classification_reproduced": legacy,
        "trace_evidence_complete": evidence_complete,
        "trace_execution_complete": execution_complete,
        "history_cap_satisfied": isinstance(actions, list) and len(actions) <= 240,
        "collision_free": collisions == 0,
        "isolation_on_saved_observations": bool(own_hits) and not other_hits,
        "history_action_count": len(actions) if isinstance(actions, list) else None,
        "history_observation_count": len(observations),
        "first_contamination_stage": "UNRESOLVED" if other_hits else "NOT_APPLICABLE",
    }


def isolation_audit(
    reader: SnapshotReader,
    old_audit: dict[str, Any],
    proposals: list[dict[str, Any]],
    house: dict[str, Any],
    compiler_class: type,
) -> dict[str, Any]:
    proposal_by_id = {str(row["id"]): row for row in proposals}
    rows: list[dict[str, Any]] = []
    joint = Counter()
    legacy_counts = Counter()
    history_count = 0
    notice_count = 0
    for old in old_audit.get("rows", []):
        evidence_path = old.get("evidence_path")
        evidence_hash = old.get("evidence_sha256")
        is_notice = old.get("history_id") is None
        base: dict[str, Any] = {
            "proposal_id": old.get("proposal"),
            "yaw": old.get("yaw"),
            "history_id": old.get("history_id"),
            "record_kind": "PARTIAL_FAMILY_NOTICE" if is_notice else "HISTORY_DISCOVERY",
            "input_object_sha256": evidence_hash,
            "source_evidence_path": evidence_path,
            "original_error_raw": old.get("original_error"),
            "family_acceptance_status": "NOT_ACCEPTED",
        }
        if is_notice:
            notice_count += 1
            base.update(
                {
                    "trace_evidence_complete": False,
                    "trace_execution_complete": False,
                    "isolation_on_saved_observations": None,
                    "first_contamination_stage": "NOT_APPLICABLE",
                    "evidence_status": "READ",
                    "legacy_classification_reproduced": "PARTIAL_FAMILY_NOT_ACCEPTED",
                }
            )
            rows.append(base)
            continue
        history_count += 1
        proposal = proposal_by_id.get(str(old.get("proposal")))
        if proposal is None:
            base.update({"evidence_status": "EVIDENCE_INVALID", "diagnostic_error": "PROPOSAL_CONTEXT_MISSING"})
            rows.append(base)
            continue
        try:
            snapshot_record = reader.record_for_path(str(evidence_path))
            if snapshot_record["object_sha256"] != evidence_hash:
                raise EvidenceInvalid("HISTORY_EXPECTED_HASH")
            trace = reader.read_json(evidence_hash)
            compiler = make_compiler(proposal, house, compiler_class)
            base.update(_trace_event_summary(trace, compiler, str(old["history_id"]), evidence_hash))
            base["evidence_status"] = "READ"
            key = f"own={int(base['own_anchor_seen'])}/other={int(base['other_anchor_seen'])}"
            joint[key] += 1
            legacy_counts[base["legacy_classification_reproduced"]] += 1
        except EvidenceMissing as exc:
            base.update({"evidence_status": "EVIDENCE_MISSING", "diagnostic_error": repr(exc)})
        except (EvidenceInvalid, KeyError, TypeError, ValueError) as exc:
            base.update({"evidence_status": "EVIDENCE_INVALID", "diagnostic_error": repr(exc)})
        rows.append(base)
    return {
        "version": "v16_rqf_offline_errata_v2_1",
        "house": house["house"],
        "split": house["split"],
        "history_record_count": history_count,
        "partial_family_notice_count": notice_count,
        "own_other_joint_distribution": dict(sorted(joint.items())),
        "legacy_classification_counts": dict(sorted(legacy_counts.items())),
        "rows": rows,
        "stage_rule": "UNRESOLVED when another anchor is seen without explicit frozen phase boundaries",
        "raw_array_integrity": "NOT_RECHECKED",
    }

