#!/usr/bin/env python3
"""Read-only audit of the Qwen3.8-Max provisional label projections."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


import run_mf3zp_qwen38max_pilot as pilot


OUT = pilot.OUTPUT
RESULT = OUT / "MF3ZP_QWEN38MAX_LABEL_AUDIT_V2.json"


def response_ok(value: dict, request: dict) -> bool:
    if value.get("status") != "PASS" or value.get("model") != pilot.MODEL:
        return False
    if value.get("request_id") != request["request_id"] or value.get("event_id") != request["event_id"] or value.get("prefix_step") != request["prefix_step"]:
        return False
    return not pilot.m.validate_annotation_response(
        value.get("response"), event_id=str(request["event_id"]),
        prefix_step=int(request["prefix_step"]),
        allowed_aliases=[item["alias"] for item in request["contract"]["current_candidates"]],
    )


def main() -> int:
    protocol = pilot.verify_protocol()
    parent = pilot.read_json(pilot.V2_PROTOCOL)
    events = pilot.select_events(parent)
    requests, projection_rows, deterministic_rows = pilot.build_requests(events)
    projection = {row["event_id"]: row for row in projection_rows}
    deterministic: dict[str, list[dict]] = defaultdict(list)
    for row in deterministic_rows:
        deterministic[row["event_id"]].append(row)
    response_root = OUT / "responses" / pilot.MODEL.replace(".", "_").replace("-", "_")
    by_event: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        path = response_root / f"{request['request_id']}.json"
        value = pilot.read_json(path)
        if not response_ok(value, request):
            raise RuntimeError(f"invalid response: {path}")
        present = next(bool(row["target_in_set"]) for row in deterministic[request["event_id"]] if int(row["prefix_step"]) == int(request["prefix_step"]))
        state = pilot.m.derive_semantic_state(
            value["response"],
            target_alias=projection[request["event_id"]]["alternative_alias"],
            native_alias=projection[request["event_id"]]["native_alias"],
            target_present=present,
        )
        by_event[request["event_id"]].append({"prefix_step": int(request["prefix_step"]), **state})
    event_results = {}
    for event_id, rows in by_event.items():
        rows.sort(key=lambda row: row["prefix_step"])
        target = tuple(bool(row["target_in_set"]) for row in deterministic[event_id])
        separated = tuple(bool(row["candidate_separated"]) for row in rows)
        closed = tuple(bool(row["evidence_closed"]) for row in rows)
        states = pilot.m.derive_uad(target, separated, closed, stability_k=pilot.m.STABILITY_K)
        reveal = pilot.m.reveal_interval(states)
        expiry = max((int(row["prefix_step"]) for row in deterministic[event_id] if row["target_in_set"]), default=None)
        event_results[event_id] = {
            "final_state": states[-1], "states": list(states),
            "reveal_interval": reveal, "expiry_step": expiry,
            "complete": reveal is not None and expiry is not None,
            "any_evidence_closed": any(closed),
        }
    result = {
        "schema_version": "revealnav-mf3zp-qwen38max-label-audit/2",
        "status": "LABEL_READINESS_PASS" if all(row["complete"] for row in event_results.values()) else "LABEL_READINESS_INCOMPLETE",
        "model": pilot.MODEL,
        "event_count": len(event_results), "request_count": len(requests),
        "domain_counts": dict(Counter(row["dataset"] for row in events)),
        "final_state_counts": dict(Counter(row["final_state"] for row in event_results.values())),
        "complete_events": sum(bool(row["complete"]) for row in event_results.values()),
        "events_with_any_evidence_closed": sum(bool(row["any_evidence_closed"]) for row in event_results.values()),
        "event_results": event_results,
        "target_payload_read": False, "outcome_payload_read": False,
        "public_split_access": False, "scientific_probe_executed": False,
        "note": "Qwen3.8-Max responses are provisional machine annotations; no exact outcome was opened",
    }
    pilot.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
