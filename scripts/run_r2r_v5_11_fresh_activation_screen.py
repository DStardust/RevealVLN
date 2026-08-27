#!/usr/bin/env python3
"""Blind activation extension for the V5.11 temporal-history fix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_v5_6_fresh_seen_screen as v56  # noqa: E402
import run_r2r_v5_10_fresh_activation_screen as base  # noqa: E402
import run_r2r_v5_11_temporal_diagnostic as diagnostic  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_temporal_native_control_opp_worker_v5_11.py"
DIAGNOSTIC_PROTOCOL = diagnostic.PROTOCOL
DIAGNOSTIC_RESULT = diagnostic.RESULT
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_11_fresh_activation_screen"
PROTOCOL = OUT / "R2R_V5_11_FRESH_ACTIVATION_PROTOCOL.json"
RESULT = OUT / "R2R_V5_11_FRESH_ACTIVATION_RESULT.json"
TARGET_ACTIVE = base.TARGET_ACTIVE
TARGET_SCENES = base.TARGET_SCENES


def configure_base() -> None:
    base.diagnostic = diagnostic
    base.WORKER = WORKER
    base.OUT = OUT
    base.PROTOCOL = PROTOCOL
    base.RESULT = RESULT
    base.protocol_value = protocol_value
    base.diagnostic_state = diagnostic_state


def diagnostic_state() -> tuple[dict, dict[str, dict]]:
    result = json.loads(DIAGNOSTIC_RESULT.read_text())
    if not (
        result.get("status") == "V5_11_TEMPORAL_DIAGNOSTIC_PASS"
        and all(result.get("engineering_gates", {}).values())
        and result.get("task_metric_payload_read") is False
    ):
        raise RuntimeError("completed V5.11 blind diagnostic is required")
    summaries = {}
    for path in (diagnostic.OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        summaries[str(row["episode_id"])] = row
    return result, summaries


def protocol_value() -> dict:
    result = json.loads(DIAGNOSTIC_RESULT.read_text())
    if not (
        result.get("status") == "V5_11_TEMPORAL_DIAGNOSTIC_PASS"
        and all(result.get("engineering_gates", {}).values())
        and result.get("task_metric_payload_read") is False
    ):
        raise RuntimeError("completed V5.11 blind diagnostic is required")
    diagnostic_protocol = json.loads(DIAGNOSTIC_PROTOCOL.read_text())
    initial_ids = [row["episode_id"] for row in diagnostic_protocol["selection"]]
    v56_protocol = json.loads(v56.PROTOCOL.read_text())
    if initial_ids != [row["episode_id"] for row in v56_protocol["eligible"][:160]]:
        raise RuntimeError("V5.11 diagnostic/base ordering drift")
    return {
        "schema_version": "revealnav-r2r-v5.11-fresh-activation-protocol/1",
        "status": "SEALED_BEFORE_V5_11_BLIND_ACTIVATION_EXTENSION",
        "initial_diagnostic_episode_ids": initial_ids,
        "initial_active": result["active_episodes"],
        "extension_eligible": v56_protocol["eligible"][160:],
        "screen_seed": v56_protocol["screen_seed"],
        "stopping_rule": {
            "target_active_combined": TARGET_ACTIVE,
            "target_distinct_scenes_combined": TARGET_SCENES,
            "stop_checked_after_each_completed_episode": True,
        },
        "active_definition": (
            "effective_commit_interventions + explore_decisions > 0"
        ),
        "cohort_selection_rule": (
            "take the earliest active episode from each new scene until 15 "
            "scenes, then fill to 24 in the original sealed order"
        ),
        "selection_uses_task_metrics": False,
        "worker_reads_task_metric_payload": False,
        "sources": {
            str(RUNNER.relative_to(ROOT)): v56.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): v56.sha256_file(WORKER),
            str(DIAGNOSTIC_PROTOCOL.relative_to(ROOT)): v56.sha256_file(
                DIAGNOSTIC_PROTOCOL
            ),
            str(DIAGNOSTIC_RESULT.relative_to(ROOT)): v56.sha256_file(
                DIAGNOSTIC_RESULT
            ),
            str(v56.PROTOCOL.relative_to(ROOT)): v56.sha256_file(v56.PROTOCOL),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.11 activation protocol drift")
    if not PROTOCOL.exists():
        v56.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "initial_active": value["initial_active"],
        "extension_eligible": len(value["extension_eligible"]),
        "sha256": v56.sha256_file(PROTOCOL),
    }))


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.11 activation protocol must be sealed")
    _, initial = base.diagnostic_state()
    runs = OUT / "runs"
    if runs.exists() and not resume:
        raise RuntimeError("V5.11 activation runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    extension = {}
    for path in runs.glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "PASS":
            extension[str(row["episode_id"])] = row
    queue = []
    for row in protocol["extension_eligible"]:
        if row["episode_id"] in extension:
            continue
        run_dir = runs / f"shadow_ep_{row['episode_id']}"
        if run_dir.exists():
            destination = OUT / "interrupted" / f"{run_dir.name}_{int(time.time())}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(run_dir, destination)
        queue.append(row)
    free = list(gpus)
    jobs = []
    failures = []
    target_met = False
    while (queue and not target_met) or jobs:
        while queue and free and not target_met:
            row = queue.pop(0)
            gpu = free.pop(0)
            jobs.append(base.run_one(row, gpu, protocol["screen_seed"]))
        time.sleep(0.5)
        for job in list(jobs):
            code = job["process"].poll()
            if code is None:
                continue
            for stream in job["streams"]:
                stream.close()
            path = runs / job["job"] / "RUN_SUMMARY.json"
            if code or not path.is_file():
                failures.append({"job": job["job"], "returncode": code})
            else:
                row = json.loads(path.read_text())
                if row.get("status") != "PASS":
                    failures.append({"job": job["job"], "returncode": code})
                else:
                    extension[job["row"]["episode_id"]] = row
            print(json.dumps({
                "job": job["job"], "gpu": job["gpu"], "returncode": code,
            }), flush=True)
            free.append(job["gpu"])
            free.sort()
            jobs.remove(job)
            active_rows = base.combined_active(protocol, initial, extension)
            target_met = (
                len(active_rows) >= TARGET_ACTIVE
                and len({row["scene_id"] for row in active_rows}) >= TARGET_SCENES
            )
            v56.atomic_json(OUT / "RUN_STATUS.json", {
                "status": "FAIL" if failures else (
                    "DRAINING" if target_met and jobs else
                    "COMPLETE" if target_met else "RUNNING"
                ),
                "extension_completed": len(extension),
                "extension_eligible": len(protocol["extension_eligible"]),
                "combined_active": len(active_rows),
                "combined_active_scenes": len({row["scene_id"] for row in active_rows}),
                "in_flight": len(jobs),
                "failures": failures,
            })
            if failures:
                for running in jobs:
                    running["process"].terminate()
                raise RuntimeError("V5.11 activation worker failure")
    active_rows = base.combined_active(protocol, initial, extension)
    v56.atomic_json(OUT / "RUN_STATUS.json", {
        "status": "COMPLETE",
        "extension_completed": len(extension),
        "extension_eligible": len(protocol["extension_eligible"]),
        "combined_active": len(active_rows),
        "combined_active_scenes": len({row["scene_id"] for row in active_rows}),
        "in_flight": 0,
        "failures": [],
        "stopped_by_predeclared_rule": bool(queue),
    })


def verify() -> None:
    base.verify()
    result = json.loads(RESULT.read_text())
    if result.get("status") != "V5_10_FRESH_COHORT_READY":
        raise RuntimeError("shared V5.11 activation verification failed")
    result["schema_version"] = "revealnav-r2r-v5.11-fresh-activation-result/1"
    result["status"] = "V5_11_FRESH_COHORT_READY"
    v56.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU indices")
    if args.command == "seal":
        seal()
    elif args.command in ("run", "resume"):
        execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
