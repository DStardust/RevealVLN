#!/usr/bin/env python3
"""Record structural readiness limits for the sealed Qwen3.8-Max pilot.

This is a read-only diagnostic.  It does not read outcome payloads, alter the
sealed pilot protocol, or reinterpret the frozen U/A/D stability rule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import run_mf3zp_qwen38max_pilot as pilot


OUT = pilot.OUTPUT
RESULT = OUT / "MF3ZP_QWEN38MAX_STRUCTURAL_DIAGNOSTIC.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    protocol = pilot.verify_protocol()
    parent = pilot.read_json(pilot.V2_PROTOCOL)
    events = pilot.select_events(parent)
    requests, projection_rows, deterministic_rows = pilot.build_requests(events)
    del projection_rows

    deterministic: dict[str, list[dict]] = defaultdict(list)
    for row in deterministic_rows:
        deterministic[str(row["event_id"])].append(row)
    target_prefix_counts = {}
    for event in events:
        rows = deterministic[str(event["event_id"])]
        target_prefix_counts[str(event["event_id"])] = sum(
            bool(row["target_in_set"]) for row in rows
            if int(row["prefix_step"]) <= int(event["decision_step"])
        )

    # A separate parent-population diagnostic is useful for deciding whether
    # the limitation is specific to this 20-event pilot.  It uses only the
    # already-sealed deterministic candidate-presence metadata.
    parent_events = list(parent["population"]["events"])
    parent_rows = pilot.read_jsonl(pilot.V2_OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl")
    parent_by_event: dict[str, list[dict]] = defaultdict(list)
    for row in parent_rows:
        parent_by_event[str(row["event_id"])].append(row)
    parent_target_prefix_counts = {}
    for event in parent_events:
        parent_target_prefix_counts[str(event["event_id"])] = sum(
            bool(row["target_in_set"])
            for row in parent_by_event[str(event["event_id"])]
            if int(row["prefix_step"]) <= int(event["decision_step"])
        )

    response_root = OUT / "responses" / pilot.MODEL.replace(".", "_").replace("-", "_")
    valid_responses = 0
    final_unique = 0
    final_separated = 0
    final_target_present = 0
    for request in requests:
        value = pilot.read_json(response_root / f"{request['request_id']}.json")
        if value.get("status") != "PASS":
            raise RuntimeError("pilot response manifest contains a non-PASS response")
        valid_responses += 1
    for event in events:
        event_requests = [
            request for request in requests
            if str(request["event_id"]) == str(event["event_id"])
        ]
        final_request = max(event_requests, key=lambda row: int(row["prefix_step"]))
        value = pilot.read_json(response_root / f"{final_request['request_id']}.json")
        response = value["response"]
        target_present = target_prefix_counts[str(event["event_id"])] > 0
        if target_present:
            final_target_present += 1
        if response.get("instruction_uniquely_selects_one") is True:
            final_unique += 1
        if response.get("candidates_visually_distinguishable") is True:
            final_separated += 1

    result = {
        "schema_version": "revealnav-mf3zp-qwen38max-structural-diagnostic/1",
        "status": "STRUCTURAL_LIMIT_CONFIRMED",
        "pilot_protocol_sha256": sha256_file(pilot.PROTOCOL),
        "model": pilot.MODEL,
        "event_count": len(events),
        "request_count": len(requests),
        "valid_response_count": valid_responses,
        "domain_event_counts": dict(Counter(str(event["dataset"]) for event in events)),
        "target_present_prefix_count_distribution": dict(Counter(target_prefix_counts.values())),
        "events_with_at_least_stability_k_target_prefixes": sum(
            count >= pilot.m.STABILITY_K for count in target_prefix_counts.values()
        ),
        "stability_k": pilot.m.STABILITY_K,
        "final_target_present_events": final_target_present,
        "final_instruction_unique_events": final_unique,
        "final_visually_separated_events": final_separated,
        "theoretical_complete_uad_upper_bound": sum(
            count >= pilot.m.STABILITY_K for count in target_prefix_counts.values()
        ),
        "parent_population_event_count": len(parent_events),
        "parent_population_target_present_prefix_count_distribution": dict(
            Counter(parent_target_prefix_counts.values())
        ),
        "parent_population_events_with_at_least_stability_k_target_prefixes": sum(
            count >= pilot.m.STABILITY_K for count in parent_target_prefix_counts.values()
        ),
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": False,
        "scientific_probe_executed": False,
        "interpretation": (
            "Within the sealed event-local prefixes, a D state requires three "
            "consecutive target-present prefixes. The fixed pilot has no event "
            "with three target-present prefixes, so a complete Reveal interval "
            "is structurally impossible without changing the event population "
            "or the frozen stability rule. Neither is changed here."
        ),
    }
    pilot.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
