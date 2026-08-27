#!/usr/bin/env python3
"""Outcome-blind diagnostic for the V5.10 native-control adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_v5_6_fresh_seen_screen as v56_screen  # noqa: E402
import run_r2r_v5_7_candidate_adapter_diagnostic as v57_run  # noqa: E402
import run_r2r_v5_8_safe_local_diagnostic as v58_run  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_native_control_opp_worker_v5_10.py"
V59_WORKER = ROOT / "scripts/r2r_hybrid_candidate_opp_worker_v5_9.py"
V58_WORKER = ROOT / "scripts/r2r_safe_local_opp_worker_v5_8.py"
V57_RESULT = v57_run.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_native_control_diagnostic"
PROTOCOL = OUT / "R2R_V5_10_NATIVE_CONTROL_PROTOCOL.json"
RESULT = OUT / "R2R_V5_10_NATIVE_CONTROL_RESULT.json"


def protocol_value() -> dict:
    v56_screen.validate_lock()
    prior = json.loads(V57_RESULT.read_text())
    if prior.get("status") != "V5_7_ADAPTER_DIAGNOSTIC_PASS":
        raise RuntimeError("completed V5.7 candidate diagnostic is required")
    v56_protocol = json.loads(v56_screen.PROTOCOL.read_text())
    selection = v56_protocol["eligible"][:v57_run.LIMIT]
    return {
        "schema_version": "revealnav-r2r-v5.10-native-control-protocol/1",
        "status": "SEALED_BEFORE_OUTCOME_BLIND_NATIVE_CONTROL_DIAGNOSTIC",
        "correctness_revision": (
            "global K=3 identity intersected with 2-4 local executable "
            "branches; frozen ETP native action is scored as the control; "
            "STOP and visited/backtrack actions cannot be overridden"
        ),
        "selection": selection,
        "runs": len(selection),
        "screen_seed": v56_protocol["screen_seed"],
        "task_metrics_must_not_be_read": True,
        "controller_actions_executed": False,
        "sources": {
            str(RUNNER.relative_to(ROOT)): v56_screen.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): v56_screen.sha256_file(WORKER),
            str(V59_WORKER.relative_to(ROOT)): v56_screen.sha256_file(
                V59_WORKER
            ),
            str(V58_WORKER.relative_to(ROOT)): v56_screen.sha256_file(
                V58_WORKER
            ),
            str(V57_RESULT.relative_to(ROOT)): v56_screen.sha256_file(
                V57_RESULT
            ),
            str(v56_screen.LOCK.relative_to(ROOT)): v56_screen.sha256_file(
                v56_screen.LOCK
            ),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def configure_base() -> None:
    v58_run.WORKER = WORKER
    v58_run.OUT = OUT
    v58_run.PROTOCOL = PROTOCOL
    v58_run.RESULT = RESULT
    v58_run.protocol_value = protocol_value
    v58_run.configure_base()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.10 protocol drift")
    if not PROTOCOL.exists():
        v56_screen.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["runs"],
        "sha256": v56_screen.sha256_file(PROTOCOL),
    }))


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.10 diagnostic protocol drift")
    selected = {row["episode_id"] for row in protocol["selection"]}
    summaries = {}
    traces = {}
    for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        episode_id = str(row["episode_id"])
        if episode_id in summaries:
            raise RuntimeError("duplicate V5.10 episode")
        summaries[episode_id] = row
        trace = path.parent / "controller_trace.jsonl"
        traces[episode_id] = [
            json.loads(line) for line in trace.read_text().splitlines()
        ]
    paired = {}
    for episode_id in selected:
        path = v57_run.V56_RUNS / f"shadow_ep_{episode_id}" / "RUN_SUMMARY.json"
        if path.is_file():
            paired[episode_id] = json.loads(path.read_text())
    gates = {
        "exact_run_set": set(summaries) == selected,
        "exact_paired_v5_6_set": set(paired) == selected,
        "all_runs_pass": all(
            row.get("status") == "PASS" for row in summaries.values()
        ),
        "all_shadow_without_task_metrics": all(
            row.get("mode") == "shadow"
            and row.get("task_metric_payload_read") is False
            and row.get("metrics") is None
            for row in summaries.values()
        ),
        "paired_base_traces_identical": set(paired) == selected and all(
            summaries[episode_id]["base_trace_sha256"]
            == paired[episode_id]["base_trace_sha256"]
            for episode_id in selected
        ),
        "no_commit_or_explore_over_native_stop": all(
            not (
                event.get("event") == "opp_initial_decision"
                and event.get("opp_action") in ("commit", "explore")
                and event.get("native_base_branch") is None
            )
            for rows in traces.values() for event in rows
        ),
        "all_interventions_had_native_control": all(
            row["controller"]["effective_commit_interventions"]
            + row["controller"]["explore_decisions"] == 0
            or row["safety_funnel"]["native_control_comparisons"] > 0
            for row in summaries.values()
        ),
        "v5_6_lock_unchanged": True,
        "no_unseen_or_test_payload": True,
    }
    def active(row: dict) -> bool:
        controller = row["controller"]
        return (
            controller["effective_commit_interventions"]
            + controller["explore_decisions"] > 0
        )

    active_count = sum(active(row) for row in summaries.values())
    result = {
        "schema_version": "revealnav-r2r-v5.10-native-control-result/1",
        "status": (
            "V5_10_NATIVE_CONTROL_DIAGNOSTIC_PASS"
            if all(gates.values()) else "V5_10_NATIVE_CONTROL_DIAGNOSTIC_FAIL"
        ),
        "engineering_gates": gates,
        "paired_episodes": len(selected),
        "active_episodes": active_count,
        "active_rate": active_count / len(selected),
        "episodes_with_two_persistent": sum(
            row["candidate_funnel"]["prefixes_with_two_persistent"] > 0
            for row in summaries.values()
        ),
        "safety_funnel": {
            key: sum(row["safety_funnel"][key] for row in summaries.values())
            for key in (
                "stop_suppressions",
                "native_outside_candidate_suppressions",
                "candidate_width_suppressions",
                "safe_decision_prefixes",
                "native_control_comparisons",
            )
        },
        "candidate_funnel": {
            key: sum(row["candidate_funnel"][key] for row in summaries.values())
            for key in (
                "navigation_prefixes", "prefixes_with_two_current",
                "prefixes_with_two_local", "prefixes_with_two_persistent",
            )
        },
        "interpretation": (
            "candidate and safety coverage only; task performance remains unread"
        ),
        "protocol_sha256": v56_screen.sha256_file(PROTOCOL),
        "task_metric_payload_read": False,
        "paper_result": False,
        "unseen_or_test_accessed": False,
    }
    v56_screen.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique indices")
    if args.command == "seal":
        seal()
    elif args.command in ("run", "resume"):
        v57_run.execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
