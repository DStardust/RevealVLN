#!/usr/bin/env python3
"""Extend the blind V5.10 diagnostic until a diverse active cohort exists."""

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

import run_r2r_v5_6_fresh_seen_screen as v56  # noqa: E402
import run_r2r_v5_10_native_control_diagnostic as diagnostic  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_native_control_opp_worker_v5_10.py"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_10_fresh_activation_screen"
PROTOCOL = OUT / "R2R_V5_10_FRESH_ACTIVATION_PROTOCOL.json"
RESULT = OUT / "R2R_V5_10_FRESH_ACTIVATION_RESULT.json"
TARGET_ACTIVE = 24
TARGET_SCENES = 15


def active(summary: dict) -> bool:
    controller = summary["controller"]
    return (
        controller["effective_commit_interventions"]
        + controller["explore_decisions"] > 0
    )


def diagnostic_state() -> tuple[dict, dict[str, dict]]:
    result = json.loads(diagnostic.RESULT.read_text())
    if not (
        result.get("status") == "V5_10_NATIVE_CONTROL_DIAGNOSTIC_PASS"
        and all(result.get("engineering_gates", {}).values())
        and result.get("task_metric_payload_read") is False
    ):
        raise RuntimeError("completed V5.10 blind diagnostic is required")
    summaries = {}
    for path in (diagnostic.OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        summaries[str(row["episode_id"])] = row
    return result, summaries


def protocol_value() -> dict:
    result, initial = diagnostic_state()
    base = json.loads(v56.PROTOCOL.read_text())
    diagnostic_ids = [
        row["episode_id"]
        for row in json.loads(diagnostic.PROTOCOL.read_text())["selection"]
    ]
    if (
        diagnostic_ids != [row["episode_id"] for row in base["eligible"][:160]]
        or set(initial) != set(diagnostic_ids)
    ):
        raise RuntimeError("V5.10 diagnostic/base ordering drift")
    extension = base["eligible"][160:]
    return {
        "schema_version": "revealnav-r2r-v5.10-fresh-activation-protocol/1",
        "status": "SEALED_BEFORE_V5_10_BLIND_ACTIVATION_EXTENSION",
        "initial_diagnostic_episode_ids": diagnostic_ids,
        "initial_active": result["active_episodes"],
        "extension_eligible": extension,
        "screen_seed": base["screen_seed"],
        "stopping_rule": {
            "wave_size": 8,
            "target_active_combined": TARGET_ACTIVE,
            "target_distinct_scenes_combined": TARGET_SCENES,
            "stop_only_between_complete_waves": True,
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
            str(diagnostic.PROTOCOL.relative_to(ROOT)): v56.sha256_file(
                diagnostic.PROTOCOL
            ),
            str(diagnostic.RESULT.relative_to(ROOT)): v56.sha256_file(
                diagnostic.RESULT
            ),
            str(v56.PROTOCOL.relative_to(ROOT)): v56.sha256_file(v56.PROTOCOL),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.10 activation protocol drift")
    if not PROTOCOL.exists():
        v56.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "initial_active": value["initial_active"],
        "extension_eligible": len(value["extension_eligible"]),
        "sha256": v56.sha256_file(PROTOCOL),
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
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }, stdout=stdout, stderr=stderr)
    return {
        "row": row, "job": job, "gpu": gpu, "process": process,
        "streams": (stdout, stderr),
    }


def combined_active(
    protocol: dict, initial: dict[str, dict], extension: dict[str, dict],
) -> list[dict]:
    metadata = {
        row["episode_id"]: row
        for row in json.loads(v56.PROTOCOL.read_text())["eligible"]
    }
    ordered = [
        *protocol["initial_diagnostic_episode_ids"],
        *[row["episode_id"] for row in protocol["extension_eligible"]],
    ]
    summaries = {**initial, **extension}
    return [
        metadata[episode_id] for episode_id in ordered
        if episode_id in summaries and active(summaries[episode_id])
    ]


def select_cohort(active_rows: list[dict]) -> list[dict]:
    selected = []
    scenes = set()
    for row in active_rows:
        if row["scene_id"] not in scenes and len(scenes) < TARGET_SCENES:
            selected.append(row)
            scenes.add(row["scene_id"])
    selected_ids = {row["episode_id"] for row in selected}
    for row in active_rows:
        if len(selected) >= TARGET_ACTIVE:
            break
        if row["episode_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["episode_id"])
    return selected


def execute(gpus: tuple[int, ...], resume: bool) -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.10 activation protocol must be sealed")
    _, initial = diagnostic_state()
    runs = OUT / "runs"
    if runs.exists() and not resume:
        raise RuntimeError("V5.10 activation runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    extension = {}
    for path in runs.glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "PASS":
            extension[str(row["episode_id"])] = row
    queue = [
        row for row in protocol["extension_eligible"]
        if row["episode_id"] not in extension
    ]
    while queue:
        active_rows = combined_active(protocol, initial, extension)
        if (
            len(active_rows) >= TARGET_ACTIVE
            and len({row["scene_id"] for row in active_rows}) >= TARGET_SCENES
        ):
            break
        wave, queue = queue[:len(gpus)], queue[len(gpus):]
        jobs = [
            run_one(row, gpu, protocol["screen_seed"])
            for row, gpu in zip(wave, gpus)
        ]
        failures = []
        for job in jobs:
            code = job["process"].wait()
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
        active_rows = combined_active(protocol, initial, extension)
        v56.atomic_json(OUT / "RUN_STATUS.json", {
            "status": "FAIL" if failures else "RUNNING",
            "extension_completed": len(extension),
            "extension_eligible": len(protocol["extension_eligible"]),
            "combined_active": len(active_rows),
            "combined_active_scenes": len({row["scene_id"] for row in active_rows}),
            "failures": failures,
        })
        if failures:
            raise RuntimeError("V5.10 activation worker failure")
    active_rows = combined_active(protocol, initial, extension)
    v56.atomic_json(OUT / "RUN_STATUS.json", {
        "status": "COMPLETE",
        "extension_completed": len(extension),
        "extension_eligible": len(protocol["extension_eligible"]),
        "combined_active": len(active_rows),
        "combined_active_scenes": len({row["scene_id"] for row in active_rows}),
        "failures": [],
        "stopped_by_predeclared_rule": bool(queue),
    })


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("V5.10 activation protocol drift")
    _, initial = diagnostic_state()
    extension = {}
    eligible = {row["episode_id"] for row in protocol["extension_eligible"]}
    for path in (OUT / "runs").glob("*/RUN_SUMMARY.json"):
        row = json.loads(path.read_text())
        episode_id = str(row["episode_id"])
        if episode_id in extension or episode_id not in eligible:
            raise RuntimeError("invalid V5.10 extension episode")
        if not (
            row.get("status") == "PASS"
            and row.get("mode") == "shadow"
            and row.get("task_metric_payload_read") is False
            and row.get("metrics") is None
        ):
            raise RuntimeError("V5.10 extension read a task metric or failed")
        extension[episode_id] = row
    active_rows = combined_active(protocol, initial, extension)
    selected = select_cohort(active_rows)
    extension_order = [row["episode_id"] for row in protocol["extension_eligible"]]
    gates = {
        "at_least_target_active": len(active_rows) >= TARGET_ACTIVE,
        "at_least_target_scenes": len({row["scene_id"] for row in active_rows}) >= TARGET_SCENES,
        "selected_count_exact": len(selected) == TARGET_ACTIVE,
        "selected_scene_count_sufficient": len({row["scene_id"] for row in selected}) >= TARGET_SCENES,
        "extension_is_exact_ordered_prefix": set(extension) == set(
            extension_order[:len(extension)]
        ),
        "all_shadow_no_task_metrics": True,
        "locked_sources_unchanged": True,
        "no_unseen_or_test_payload": True,
    }
    result = {
        "schema_version": "revealnav-r2r-v5.10-fresh-activation-result/1",
        "status": (
            "V5_10_FRESH_COHORT_READY" if all(gates.values())
            else "V5_10_FRESH_COHORT_FAIL"
        ),
        "initial_screened": len(initial),
        "extension_screened": len(extension),
        "combined_active": len(active_rows),
        "combined_active_scenes": len({row["scene_id"] for row in active_rows}),
        "selected_confirmation_cohort": selected,
        "gates": gates,
        "selection_used_task_metrics": False,
        "task_metric_payload_read": False,
        "protocol_sha256": v56.sha256_file(PROTOCOL),
        "paper_result": False,
        "unseen_or_test_accessed": False,
    }
    v56.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
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
