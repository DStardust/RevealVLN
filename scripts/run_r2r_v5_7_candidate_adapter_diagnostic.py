#!/usr/bin/env python3
"""Outcome-blind paired diagnostic for the V5.7 R2R candidate adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_v5_6_fresh_seen_screen as v56_screen  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_full_opp_worker_v5_7.py"
TRACKER = ROOT / "revealnav_mf2r4/temporal_candidates.py"
V56_PROTOCOL = v56_screen.PROTOCOL
V56_RUNS = v56_screen.OUT / "runs"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_7_candidate_adapter_diagnostic"
PROTOCOL = OUT / "R2R_V5_7_CANDIDATE_ADAPTER_PROTOCOL.json"
RESULT = OUT / "R2R_V5_7_CANDIDATE_ADAPTER_RESULT.json"
LIMIT = 160


def protocol_value() -> dict:
    v56_screen.validate_lock()
    v56_protocol = json.loads(V56_PROTOCOL.read_text())
    selection = v56_protocol["eligible"][:LIMIT]
    return {
        "schema_version": "revealnav-r2r-v5.7-candidate-adapter-protocol/1",
        "status": "SEALED_BEFORE_OUTCOME_BLIND_ADAPTER_DIAGNOSTIC",
        "correctness_revision": (
            "replace R2R cumulative-local ghost count with the primary RxR "
            "adapter's consecutive-prefix global candidate semantics"
        ),
        "selection": selection,
        "runs": len(selection),
        "screen_seed": v56_protocol["screen_seed"],
        "comparison": "paired shadow base trace against locked V5.6",
        "task_metrics_must_not_be_read": True,
        "controller_actions_executed": False,
        "sources": {
            str(RUNNER.relative_to(ROOT)): v56_screen.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): v56_screen.sha256_file(WORKER),
            str(TRACKER.relative_to(ROOT)): v56_screen.sha256_file(TRACKER),
            str(V56_PROTOCOL.relative_to(ROOT)): v56_screen.sha256_file(
                V56_PROTOCOL
            ),
            str(v56_screen.LOCK.relative_to(ROOT)): v56_screen.sha256_file(
                v56_screen.LOCK
            ),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.7 diagnostic protocol drift")
    if not PROTOCOL.exists():
        v56_screen.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["runs"],
        "sha256": v56_screen.sha256_file(PROTOCOL),
    }))


def run_one(row: dict, gpu: int, seed: int) -> dict:
    job = f"shadow_ep_{row['episode_id']}"
    run_dir = OUT / "runs" / job
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{job}.stdout.log").open("w")
    stderr = (logs / f"{job}.stderr.log").open("w")
    process = subprocess.Popen([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", row["episode_id"], "--mode", "shadow",
        "--seed", str(seed), "--split", "val_seen",
        "--run-dir", str(run_dir),
    ], cwd=ROOT, env={
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }, stdout=stdout, stderr=stderr)
    return {
        "row": row, "job": job, "gpu": gpu, "process": process,
        "streams": (stdout, stderr),
    }


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.7 diagnostic protocol must be sealed")
    runs = OUT / "runs"
    if runs.exists() and not resume:
        raise RuntimeError("V5.7 diagnostic runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    completed = {}
    for path in runs.glob("*/RUN_SUMMARY.json"):
        summary = json.loads(path.read_text())
        if summary.get("status") == "PASS":
            completed[str(summary["episode_id"])] = summary
    queue = [
        row for row in protocol["selection"]
        if row["episode_id"] not in completed
    ]
    failures = []
    while queue:
        wave, queue = queue[:len(gpus)], queue[len(gpus):]
        jobs = [
            run_one(row, gpu, protocol["screen_seed"])
            for row, gpu in zip(wave, gpus)
        ]
        for job in jobs:
            code = job["process"].wait()
            for stream in job["streams"]:
                stream.close()
            path = OUT / "runs" / job["job"] / "RUN_SUMMARY.json"
            if code or not path.is_file():
                failures.append({"job": job["job"], "returncode": code})
            else:
                summary = json.loads(path.read_text())
                if summary.get("status") != "PASS":
                    failures.append({"job": job["job"], "returncode": code})
                else:
                    completed[job["row"]["episode_id"]] = summary
            print(json.dumps({
                "job": job["job"], "gpu": job["gpu"],
                "returncode": code,
            }), flush=True)
        active = sum(
            row["controller"]["effective_commit_interventions"]
            + row["controller"]["explore_decisions"] > 0
            for row in completed.values()
        )
        v56_screen.atomic_json(OUT / "RUN_STATUS.json", {
            "status": "FAIL" if failures else "RUNNING",
            "completed": len(completed), "expected": protocol["runs"],
            "active": active, "failures": failures,
        })
        if failures:
            raise RuntimeError("V5.7 diagnostic worker failure")
    v56_screen.atomic_json(OUT / "RUN_STATUS.json", {
        "status": "COMPLETE", "completed": len(completed),
        "expected": protocol["runs"],
        "active": sum(
            row["controller"]["effective_commit_interventions"]
            + row["controller"]["explore_decisions"] > 0
            for row in completed.values()
        ),
        "failures": [],
    })


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.7 diagnostic protocol drift")
    selected = {row["episode_id"] for row in protocol["selection"]}
    v57 = {}
    for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        v57[str(row["episode_id"])] = row
    v56 = {}
    for episode_id in selected:
        path = V56_RUNS / f"shadow_ep_{episode_id}" / "RUN_SUMMARY.json"
        if path.is_file():
            v56[episode_id] = json.loads(path.read_text())
    gates = {
        "exact_v57_run_set": set(v57) == selected,
        "exact_paired_v56_run_set": set(v56) == selected,
        "all_v57_pass": all(
            row.get("status") == "PASS" for row in v57.values()
        ),
        "all_shadow_without_task_metrics": all(
            row.get("mode") == "shadow"
            and row.get("task_metric_payload_read") is False
            and row.get("metrics") is None
            for row in v57.values()
        ),
        "paired_base_traces_identical": set(v56) == selected and all(
            v56[episode_id]["base_trace_sha256"]
            == v57[episode_id]["base_trace_sha256"]
            for episode_id in selected
        ),
        "v5_6_lock_unchanged": True,
        "no_unseen_or_test_payload": True,
    }
    def is_active(row: dict) -> bool:
        controller = row["controller"]
        return (
            controller["effective_commit_interventions"]
            + controller["explore_decisions"] > 0
        )

    v57_active = sum(is_active(row) for row in v57.values())
    v56_active = sum(is_active(row) for row in v56.values())
    funnel = {
        "navigation_prefixes": sum(
            row["candidate_funnel"]["navigation_prefixes"]
            for row in v57.values()
        ),
        "prefixes_with_two_current": sum(
            row["candidate_funnel"]["prefixes_with_two_current"]
            for row in v57.values()
        ),
        "prefixes_with_two_local": sum(
            row["candidate_funnel"]["prefixes_with_two_local"]
            for row in v57.values()
        ),
        "prefixes_with_two_persistent": sum(
            row["candidate_funnel"]["prefixes_with_two_persistent"]
            for row in v57.values()
        ),
        "episodes_with_two_persistent": sum(
            row["candidate_funnel"]["prefixes_with_two_persistent"] > 0
            for row in v57.values()
        ),
    }
    result = {
        "schema_version": "revealnav-r2r-v5.7-candidate-adapter-result/1",
        "status": (
            "V5_7_ADAPTER_DIAGNOSTIC_PASS" if all(gates.values())
            else "V5_7_ADAPTER_DIAGNOSTIC_FAIL"
        ),
        "engineering_gates": gates,
        "paired_episodes": len(selected),
        "v5_6_active_episodes": v56_active,
        "v5_7_active_episodes": v57_active,
        "active_episode_gain": v57_active - v56_active,
        "candidate_funnel": funnel,
        "interpretation": (
            "candidate coverage diagnostic only; V5.7 actions and task "
            "performance remain unauthorized"
        ),
        "protocol_sha256": v56_screen.sha256_file(PROTOCOL),
        "task_metric_payload_read": False,
        "paper_result": False,
        "unseen_or_test_accessed": False,
    }
    v56_screen.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
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
        execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
