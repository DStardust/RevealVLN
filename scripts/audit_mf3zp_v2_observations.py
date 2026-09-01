#!/usr/bin/env python3
"""Independent fail-closed audit for MF3ZP v2 causal observations.

This read-only gate runs before annotation assembly.  It verifies the event
identity at the decision prefix because the historical v1 assembler was not
written for prefix-witness traces.  It never opens task metrics or outcome
payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
PROTOCOL = OUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
RESULT = OUT / "MF3ZP_V2_OBSERVATION_AUDIT.json"
FORBIDDEN = {
    "target", "delta_utility", "utility", "outcome", "catastrophic",
    "treatment_result", "future_frame", "future_observation", "navmesh",
    "pose", "simulator_pose", "fold",
}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan_keys(value, path="$", violations=None):
    violations = [] if violations is None else violations
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in FORBIDDEN or any(lowered.startswith(prefix) for prefix in ("future_", "oracle_", "outcome_", "target_")):
                violations.append(f"{path}.{key}")
            scan_keys(child, f"{path}.{key}", violations)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_keys(child, f"{path}[{index}]", violations)
    return violations


def main() -> int:
    try:
        if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
            raise AuditError("v2 protocol missing")
        protocol = load_json(PROTOCOL)
        if protocol.get("revision") != "mf3zp_qwen_uad_reference_v2":
            raise AuditError("protocol revision drift")
        access = protocol.get("authorization", {}).get("public_split_access")
        if access != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
            raise AuditError("public split access is not fail-closed")
        events = protocol["population"]["events"]
        episodes = protocol["population"]["episodes"]
        by_episode = {(str(x["dataset"]), str(x["episode_id"])): x for x in episodes}
        if len(events) != 150 or len(episodes) != 149:
            raise AuditError("population cardinality drift")
        failures = []
        checked_events = 0
        checked_episodes = 0
        for key, task in by_episode.items():
            run_root = OUT / "observations" / key[0] / f"ep_{key[1]}"
            passing = []
            for summary_path in sorted(run_root.glob("attempt_*/RUN_SUMMARY.json")):
                summary = load_json(summary_path)
                if summary.get("status") == "PASS" and summary.get("source_prefix_replay_exact") is True:
                    passing.append((summary_path.parent, summary))
            if len(passing) != 1:
                failures.append({"kind": "passing_run_count", "episode": key, "count": len(passing)})
                continue
            run_dir, summary = passing[0]
            if summary.get("decision_step") != int(task["decision_step"]):
                failures.append({"kind": "decision_step_drift", "episode": key})
                continue
            record_path = run_dir / "causal_prefix_records.jsonl"
            if not record_path.is_file() or record_path.is_symlink():
                failures.append({"kind": "missing_records", "episode": key})
                continue
            causal = rows(record_path)
            if len(causal) != int(task["decision_step"]) + 1:
                failures.append({"kind": "record_count", "episode": key, "count": len(causal)})
                continue
            if scan_keys(causal):
                failures.append({"kind": "forbidden_causal_field", "episode": key})
                continue
            checked_episodes += 1
            for event in [e for e in events if (str(e["dataset"]), str(e["episode_id"])) == key]:
                step = int(event["decision_step"])
                matches = [x for x in causal if int(x.get("step", -1)) == step]
                if len(matches) != 1:
                    failures.append({"kind": "event_step_missing", "event_id": event["event_id"]})
                    continue
                observed = matches[0]
                candidates = tuple(str(x) for x in observed.get("candidate_action_ids", []))
                sealed = tuple(str(x) for x in event.get("sealed_candidate_action_ids", []))
                if observed.get("native_action_id") != event["native_action_id"]:
                    failures.append({"kind": "native_identity", "event_id": event["event_id"], "observed": observed.get("native_action_id"), "expected": event["native_action_id"]})
                    continue
                if event["native_action_id"] not in candidates or event["alternative_action_id"] not in candidates:
                    failures.append({"kind": "candidate_missing", "event_id": event["event_id"], "observed": candidates})
                    continue
                if sealed and set(candidates) != set(sealed):
                    failures.append({"kind": "candidate_set_drift", "event_id": event["event_id"], "observed": candidates, "expected": sealed})
                    continue
                checked_events += 1
        result = {
            "schema_version": "revealnav-mf3zp-v2-observation-audit/1",
            "status": "PASS" if not failures and checked_events == len(events) and checked_episodes == len(episodes) else "FAIL",
            "protocol_sha256": sha256(PROTOCOL),
            "planned_events": len(events),
            "checked_events": checked_events,
            "planned_episodes": len(episodes),
            "checked_episodes": checked_episodes,
            "failures": failures,
            "target_payload_read": False,
            "outcome_payload_read": False,
            "public_split_access": False,
        }
        RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 2
    except BaseException as error:
        print(f"MF3ZP_V2_AUDIT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
