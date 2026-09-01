#!/usr/bin/env python3
"""Audit provisional repaired Qwen labels without opening exact outcomes.

This is a read-only, fail-closed diagnostic.  It records whether the
event-local Qwen projections contain complete U/A/D -> reveal/expiry inputs
for the exploratory scout.  It never imputes unavailable fields and never
reads intervention targets or public splits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOUT = ROOT / "scripts/run_mf3zp_repaired_scout.py"
OUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1"
RESULT = OUT / "MF3ZP_REPAIRED_ORACLE_READINESS.json"


def load_scout():
    spec = importlib.util.spec_from_file_location("mf3zp_repaired_scout_readiness", SCOUT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repaired scout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    scout = load_scout()
    protocol = scout.verify_scout_protocol()
    parent_protocol = scout.read_json(scout.V2_PROTOCOL)
    requests, projection_rows, deterministic_rows = scout.build_filtered_inputs(parent_protocol)
    projection = {row["event_id"]: row for row in projection_rows}
    deterministic = defaultdict(list)
    for row in deterministic_rows:
        deterministic[row["event_id"]].append(row)
    limits = scout.event_steps(parent_protocol)
    models = [str(value) for value in parent_protocol["annotation"]["models"]]
    model_results = {}
    for model in models:
        by_event = defaultdict(list)
        for request in requests:
            _, stored = scout._resolve_response(model, request)
            state = scout.m.derive_semantic_state(
                stored["response"],
                target_alias=projection[request["event_id"]]["alternative_alias"],
                native_alias=projection[request["event_id"]]["native_alias"],
                target_present=next(
                    bool(row["target_in_set"])
                    for row in deterministic[request["event_id"]]
                    if int(row["prefix_step"]) == int(request["prefix_step"])
                ),
            )
            by_event[request["event_id"]].append((int(request["prefix_step"]), state))
        final_states = Counter()
        reveal_available = 0
        expiry_available = 0
        complete = 0
        evidence_positive = 0
        for event_id, rows in by_event.items():
            rows.sort()
            target = tuple(bool(row["target_in_set"]) for row in deterministic[event_id])
            separated = tuple(state["candidate_separated"] for _, state in rows)
            closed = tuple(state["evidence_closed"] for _, state in rows)
            states = scout.m.derive_uad(target, separated, closed, stability_k=scout.m.STABILITY_K)
            final_states[str(states[-1])] += 1
            reveal = scout.m.reveal_interval(states)
            expiry = max((int(row["prefix_step"]) for row in deterministic[event_id] if row["target_in_set"]), default=None)
            reveal_available += int(reveal is not None)
            expiry_available += int(expiry is not None)
            complete += int(reveal is not None and expiry is not None)
            evidence_positive += int(any(closed))
        model_results[model] = {
            "events": len(by_event),
            "final_state_counts": dict(final_states),
            "events_with_any_evidence_closed": evidence_positive,
            "reveal_interval_available": reveal_available,
            "expiry_available": expiry_available,
            "complete_oracle_feature_events": complete,
        }
    result = {
        "schema_version": "revealnav-mf3zp-oracle-readiness/1",
        "status": "ORACLE_LABELS_INCOMPLETE_FAIL_CLOSED",
        "event_count": len(limits),
        "filtered_request_count": len(requests),
        "event_local_rule": "prefix_step <= event.decision_step",
        "models": model_results,
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": False,
        "scout_executed": False,
        "reason": "no event has a complete reveal interval and expiry pair under the fixed K=3 derivation; unavailable fields are not imputed",
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
