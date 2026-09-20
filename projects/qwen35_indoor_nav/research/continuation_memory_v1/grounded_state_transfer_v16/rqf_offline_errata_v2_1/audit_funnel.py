"""Conservative proposal/yaw/history/stage denominators over frozen records."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from common import FIXED_YAWS, HISTORY_IDS
from snapshot_io import SnapshotReader


PHYSICAL_REASONS = {
    "COLLISION",
    "DISCOVERY_BUDGET",
    "FOLLOWER_EARLY_STOP",
    "FOLLOWER_ERROR",
    "HISTORY_EVENT_STATE_NOT_ISOLATED",
    "MISSING_REQUIRED_HISTORY_SENSITIVE_LABELS",
    "NO_ANCHOR_NEUTRAL_START_VIEW",
    "NO_PATH",
    "NO_REAL_PASS_TEACHER",
    "NO_VISIBLE_SEE2_WITNESS",
    "PERTURBATION_NO_DISPLACEMENT",
    "REPAIR_NEUTRAL_YAW_EXHAUSTED",
    "UNKNOWN_CROSS_LABEL",
    "UNKNOWN_SEMANTIC_MASK",
}


def parse_jsonl_bytes(data: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = data.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b"\n"))
    limit = len(lines) - int(incomplete_tail)
    errors: list[dict[str, Any]] = []
    for index, raw in enumerate(lines[:limit], 1):
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append({"line": index, "error": repr(exc)})
            continue
        if not isinstance(value, dict):
            errors.append({"line": index, "error": "JSONL_ROW_NOT_OBJECT"})
            continue
        rows.append(dict(value, _line=index))
    return rows, {
        "complete_rows": len(rows),
        "parse_errors": errors,
        "incomplete_tail_bytes": len(lines[-1]) if incomplete_tail else 0,
    }


def classify_legacy_result(value: Any) -> str:
    if value in {"CERTIFIED", "ACCEPTED", "PASS"}:
        return "TERMINAL_CERTIFIED_RECORD"
    if value in {"NOT_REACHED", None}:
        return "NOT_REACHED"
    if not isinstance(value, str):
        return "UNRESOLVED_LEGACY_ERROR_TYPE"
    match = re.fullmatch(r"Rejected\('([A-Z0-9_]+)'\)", value)
    if match and match.group(1) in PHYSICAL_REASONS:
        return "TERMINAL_PHYSICAL_REJECTED"
    if match:
        return "UNRESOLVED_LEGACY_ERROR_TYPE"
    return "PROGRAM_ERROR"


def _repair_path(proposal_id: str, name: str) -> str:
    return (
        "research/continuation_memory_v1/grounded_state_transfer_v16/runs/"
        f"v16_repair_rqf_001/proposals/{proposal_id}/{name}"
    )


def _family_record_complete(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("content_root"), str):
        return False
    return set(value.get("histories", {})) == set(HISTORY_IDS) and len(value.get("traces", {})) == 12


def failure_funnel(
    reader: SnapshotReader,
    proposals: list[dict[str, Any]],
    isolation: dict[str, Any],
) -> dict[str, Any]:
    proposal_ids = [str(row["id"]) for row in proposals]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("DUPLICATE_PROPOSAL_ID")
    records = reader.list_records("original_v2_input")
    paths = {record["source_path"] for record in records if record.get("object_sha256")}
    collection_path = (
        "research/continuation_memory_v1/grounded_state_transfer_v16/runs/"
        "v16_repair_rqf_001/COLLECTION_ATTEMPTS.jsonl"
    )
    collection_rows, collection_parse = parse_jsonl_bytes(reader.read_path_bytes(collection_path))
    starts = {str(row["proposal"]) for row in collection_rows if row.get("status") == "START" and row.get("proposal")}
    ends = {str(row["proposal"]) for row in collection_rows if row.get("status") in {"REJECTED", "CERTIFIED"} and row.get("proposal")}

    proposal_rows: list[dict[str, Any]] = []
    proposal_counts: Counter[str] = Counter()
    for proposal_id in proposal_ids:
        family_path = _repair_path(proposal_id, "FAMILY.json")
        failure_path = _repair_path(proposal_id, "FAILURE.json")
        if family_path in paths:
            family = reader.read_path_json(family_path)
            state = "COMPLETE_ACCEPTED_RECORD" if _family_record_complete(family) else "PARTIAL_FAMILY"
        elif failure_path in paths or proposal_id in ends:
            state = "COMPLETED_REJECTED"
        elif proposal_id in starts:
            state = "IN_PROGRESS"
        else:
            state = "NOT_REACHED"
        proposal_counts[state] += 1
        proposal_rows.append(
            {
                "proposal_id": proposal_id,
                "state": state,
                "family_object_sha256": reader.record_for_path(family_path)["object_sha256"] if family_path in paths else None,
                "failure_object_sha256": reader.record_for_path(failure_path)["object_sha256"] if failure_path in paths else None,
            }
        )

    attempts_by_slot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    attempt_parse: list[dict[str, Any]] = []
    for proposal_id in proposal_ids:
        path = _repair_path(proposal_id, "REPAIR_YAW_ATTEMPTS.jsonl")
        if path not in paths:
            continue
        attempt_rows, parse_result = parse_jsonl_bytes(reader.read_path_bytes(path))
        if parse_result["parse_errors"] or parse_result["incomplete_tail_bytes"]:
            attempt_parse.append({"proposal_id": proposal_id, "path": path, **parse_result})
        for row in attempt_rows:
            yaw = row.get("yaw")
            if type(yaw) is not int or yaw not in FIXED_YAWS:
                attempt_parse.append({"proposal_id": proposal_id, "path": path, "error": f"INVALID_YAW:{yaw!r}"})
                continue
            attempts_by_slot[(proposal_id, yaw)].append(row)

    yaw_rows: list[dict[str, Any]] = []
    yaw_counts: Counter[str] = Counter()
    identity_conflicts: list[dict[str, Any]] = []
    for proposal_id in proposal_ids:
        for yaw in FIXED_YAWS:
            attempts = attempts_by_slot.get((proposal_id, yaw), [])
            if len(attempts) > 1:
                identity_conflicts.append(
                    {"proposal_id": proposal_id, "yaw": yaw, "attempt_count": len(attempts)}
                )
                status = "EVIDENCE_UNRESOLVED"
                raw = [row.get("error", row.get("status")) for row in attempts]
            elif not attempts:
                status, raw = "NOT_REACHED", None
            else:
                attempt = attempts[0]
                raw = attempt.get("error") or attempt.get("reason") or attempt.get("status")
                status = classify_legacy_result(raw)
            yaw_counts[status] += 1
            yaw_rows.append(
                {
                    "source_run": "v16_repair_rqf_001",
                    "proposal_id": proposal_id,
                    "yaw": yaw,
                    "attempt_id": f"{proposal_id}:yaw_{yaw:02d}:attempt_1",
                    "original_yaw_status": raw,
                    "execution_status": status,
                    "publication_status": "NOT_PUBLISHED_IN_ERRATA",
                    "certificate_integrity_status": "NOT_APPLICABLE" if status != "TERMINAL_CERTIFIED_RECORD" else "REQUIRES_FAMILY_AUDIT",
                    "accepted_as_new_family": False,
                }
            )

    history_rows = [row for row in isolation.get("rows", []) if row.get("record_kind") == "HISTORY_DISCOVERY"]
    notices = [row for row in isolation.get("rows", []) if row.get("record_kind") == "PARTIAL_FAMILY_NOTICE"]
    history_counts = Counter(row.get("evidence_status", "UNKNOWN") for row in history_rows)
    proposal_identity = sum(proposal_counts.values())
    yaw_identity = sum(yaw_counts.values())
    expected_yaw_slots = len(proposal_ids) * len(FIXED_YAWS)
    return {
        "version": "v16_rqf_offline_errata_v2_1",
        "proposal_denominator": {
            "N_proposal_manifest": len(proposal_ids),
            "N_completed_rejected": proposal_counts["COMPLETED_REJECTED"],
            "N_complete_accepted_record": proposal_counts["COMPLETE_ACCEPTED_RECORD"],
            "N_partial_family": proposal_counts["PARTIAL_FAMILY"],
            "N_in_progress": proposal_counts["IN_PROGRESS"],
            "N_not_reached": proposal_counts["NOT_REACHED"],
            "N_error": proposal_counts["ERROR"],
            "N_evidence_unresolved": proposal_counts["EVIDENCE_UNRESOLVED"],
            "identity_sum": proposal_identity,
            "identity_holds": proposal_identity == len(proposal_ids),
            "old_formal_attempts_separate": 117,
        },
        "yaw_denominator": {
            "N_expected_yaw_slots": expected_yaw_slots,
            "N_terminal_physical_rejected": yaw_counts["TERMINAL_PHYSICAL_REJECTED"],
            "N_terminal_certified_record": yaw_counts["TERMINAL_CERTIFIED_RECORD"],
            "N_program_error": yaw_counts["PROGRAM_ERROR"] + yaw_counts["UNRESOLVED_LEGACY_ERROR_TYPE"],
            "N_interrupted": yaw_counts["INTERRUPTED"],
            "N_in_progress": yaw_counts["IN_PROGRESS"],
            "N_not_reached": yaw_counts["NOT_REACHED"],
            "N_evidence_unresolved": yaw_counts["EVIDENCE_UNRESOLVED"],
            "identity_sum": yaw_identity,
            "identity_holds": yaw_identity == expected_yaw_slots,
        },
        "history_denominator": {
            "history_trace_records": len(history_rows),
            "partial_family_notices_separate": len(notices),
            "evidence_status_counts": dict(sorted(history_counts.items())),
        },
        "stage_denominator": {
            "explicit_stage_boundaries": 0,
            "UNRESOLVED": sum(row.get("first_contamination_stage") == "UNRESOLVED" for row in history_rows),
            "NOT_APPLICABLE": sum(row.get("first_contamination_stage") == "NOT_APPLICABLE" for row in history_rows),
            "note": "history and yaw counts are not added to form a stage denominator",
        },
        "proposal_rows": proposal_rows,
        "yaw_rows": yaw_rows,
        "identity_conflicts": identity_conflicts,
        "parse_diagnostics": {
            "collection_log": collection_parse,
            "yaw_attempt_logs": attempt_parse,
        },
        "accepted_new_families": 0,
        "new_physical_candidate_executions": 0,
    }

