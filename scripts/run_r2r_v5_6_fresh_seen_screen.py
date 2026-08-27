#!/usr/bin/env python3
"""Outcome-blind fresh val_seen activation screen for locked V5.6."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_full_opp_worker_v5_6.py"
LOCK = ROOT / "locks/R2R_FULL_OPP_CONTROLLER_V5_6.json"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/"
    "val_seen/val_seen.json.gz"
)
PRIOR_PROTOCOLS = (
    ROOT / (
        "artifacts/evaluation/mf2_r2r_continuous_metric_v5_3_seen_active_dev/"
        "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_3.json"
    ),
    ROOT / (
        "artifacts/evaluation/mf2_r2r_continuous_metric_v5_3_seen_dev/"
        "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_3.json"
    ),
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_6_fresh_seen_screen"
PROTOCOL = OUT / "R2R_V5_6_FRESH_SEEN_SCREEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_6_FRESH_SEEN_SCREEN_RESULT.json"
SCREEN_SEED = 20260826
TARGET_ACTIVE = 30
TARGET_SCENES = 20
SALT = "revealnav-v5.6-fresh-seen-screen-20260827"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def validate_lock() -> dict:
    lock = json.loads(LOCK.read_text())
    if not (
        lock.get("status") == "LOCKED_FOR_FRESH_VAL_SEEN_CONFIRMATION"
        and lock.get("unseen_access_authorized") is False
    ):
        raise RuntimeError("V5.6 controller lock is invalid")
    for relative, evidence in lock["source_closure"].items():
        path = ROOT / relative
        if (
            path.is_symlink() or not path.is_file()
            or path.stat().st_size != evidence["bytes"]
            or sha256_file(path) != evidence["sha256"]
        ):
            raise RuntimeError(f"locked V5.6 source drift: {relative}")
    return lock


def selection() -> tuple[list[dict], list[str]]:
    excluded = set()
    for path in PRIOR_PROTOCOLS:
        payload = json.loads(path.read_text())
        excluded.update(str(row["episode_id"]) for row in payload["selection"])
    with gzip.open(DATASET, "rt") as stream:
        rows = json.load(stream)["episodes"]
    eligible = [{
        "episode_id": str(row["episode_id"]),
        "scene_id": Path(row["scene_id"]).stem,
        "trajectory_id": row.get("trajectory_id"),
    } for row in rows if str(row["episode_id"]) not in excluded]
    eligible.sort(key=lambda row: hashlib.sha256(
        f"{SALT}|{row['scene_id']}|{row['episode_id']}".encode()
    ).hexdigest())
    if len(eligible) != 734 or len(excluded) != 44:
        raise RuntimeError("fresh val_seen eligibility count drift")
    return eligible, sorted(excluded)


def protocol_value() -> dict:
    validate_lock()
    eligible, excluded = selection()
    return {
        "schema_version": "revealnav-r2r-v5.6-fresh-seen-screen-protocol/1",
        "status": "SEALED_BEFORE_FRESH_OUTCOME_BLIND_SCREEN",
        "screen_seed": SCREEN_SEED, "eligible": eligible,
        "excluded_prior_task_metric_episode_ids": excluded,
        "deterministic_order_salt": SALT,
        "stopping_rule": {
            "wave_size": 8, "target_active": TARGET_ACTIVE,
            "target_distinct_scenes": TARGET_SCENES,
            "stop_only_between_complete_waves": True,
        },
        "active_definition": (
            "effective_commit_interventions + explore_decisions > 0"
        ),
        "selection_uses_task_metrics": False,
        "worker_reads_task_metric_payload": False,
        "sources": {
            str(LOCK.relative_to(ROOT)): sha256_file(LOCK),
            str(DATASET.relative_to(ROOT)): sha256_file(DATASET),
            **{
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in PRIOR_PROTOCOLS
            },
        },
        "paper_result": False, "unseen_or_test_allowed": False,
    }


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed fresh screen protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "eligible": len(value["eligible"]),
        "sha256": sha256_file(PROTOCOL),
    }))


def active(summary: dict) -> bool:
    controller = summary["controller"]
    return (
        controller["effective_commit_interventions"]
        + controller["explore_decisions"] > 0
    )


def run_one(row: dict, gpu: int) -> dict:
    job = f"shadow_ep_{row['episode_id']}"
    run_dir = OUT / "runs" / job
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{job}.stdout.log").open("w")
    stderr = (logs / f"{job}.stderr.log").open("w")
    process = subprocess.Popen([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", row["episode_id"], "--mode", "shadow",
        "--seed", str(SCREEN_SEED), "--split", "val_seen",
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
        raise RuntimeError("fresh screen protocol must be sealed")
    runs = OUT / "runs"
    if runs.exists() and not resume:
        raise RuntimeError("fresh screen runs exist; use resume")
    runs.mkdir(parents=True, exist_ok=True)
    completed = {}
    for path in runs.glob("*/RUN_SUMMARY.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "PASS":
            completed[str(value["episode_id"])] = value
    queue = [
        row for row in protocol["eligible"]
        if row["episode_id"] not in completed
    ]
    while queue:
        active_rows = [
            row for row in protocol["eligible"]
            if row["episode_id"] in completed and active(completed[row["episode_id"]])
        ]
        if (
            len(active_rows) >= TARGET_ACTIVE
            and len({row["scene_id"] for row in active_rows}) >= TARGET_SCENES
        ):
            break
        wave = queue[:len(gpus)]
        queue = queue[len(wave):]
        jobs = [run_one(row, gpu) for row, gpu in zip(wave, gpus)]
        failures = []
        for job in jobs:
            code = job["process"].wait()
            for stream in job["streams"]:
                stream.close()
            summary_path = OUT / "runs" / job["job"] / "RUN_SUMMARY.json"
            if code or not summary_path.is_file():
                failures.append({"job": job["job"], "returncode": code})
            else:
                summary = json.loads(summary_path.read_text())
                if summary.get("status") != "PASS":
                    failures.append({"job": job["job"], "returncode": code})
                else:
                    completed[job["row"]["episode_id"]] = summary
            print(json.dumps({
                "job": job["job"], "gpu": job["gpu"], "returncode": code,
            }), flush=True)
        active_rows = [
            row for row in protocol["eligible"]
            if row["episode_id"] in completed and active(completed[row["episode_id"]])
        ]
        atomic_json(OUT / "RUN_STATUS.json", {
            "status": "FAIL" if failures else "RUNNING",
            "completed": len(completed), "eligible": len(protocol["eligible"]),
            "active": len(active_rows),
            "active_scenes": len({row["scene_id"] for row in active_rows}),
            "failures": failures,
        })
        if failures:
            raise RuntimeError("fresh screen worker failure")
    atomic_json(OUT / "RUN_STATUS.json", {
        "status": "COMPLETE",
        "completed": len(completed), "eligible": len(protocol["eligible"]),
        "active": sum(active(value) for value in completed.values()),
        "active_scenes": len({
            row["scene_id"] for row in protocol["eligible"]
            if row["episode_id"] in completed and active(completed[row["episode_id"]])
        }),
        "failures": [], "stopped_by_predeclared_rule": bool(queue),
    })


def verify() -> None:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("fresh screen protocol drift")
    metadata = {row["episode_id"]: row for row in protocol["eligible"]}
    summaries = {}
    for path in sorted((OUT / "runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        episode_id = str(row["episode_id"])
        if episode_id in summaries or episode_id not in metadata:
            raise RuntimeError("invalid fresh screen episode identity")
        if not (
            row.get("status") == "PASS" and row.get("mode") == "shadow"
            and row.get("task_metric_payload_read") is False
            and row.get("metrics") is None
        ):
            raise RuntimeError("fresh screen read a task metric or failed")
        summaries[episode_id] = row
    active_rows = [
        metadata[episode_id] for episode_id, row in summaries.items()
        if active(row)
    ]
    active_rows.sort(key=lambda row: protocol["eligible"].index(row))
    selected = active_rows[:TARGET_ACTIVE]
    gates = {
        "at_least_target_active": len(active_rows) >= TARGET_ACTIVE,
        "at_least_target_scenes": len({row["scene_id"] for row in active_rows}) >= TARGET_SCENES,
        "selected_count_exact": len(selected) == TARGET_ACTIVE,
        "all_shadow_no_task_metrics": True,
        "no_prior_metric_episode_overlap": not (
            {row["episode_id"] for row in selected}
            & set(protocol["excluded_prior_task_metric_episode_ids"])
        ),
        "locked_sources_unchanged": True,
    }
    result = {
        "schema_version": "revealnav-r2r-v5.6-fresh-seen-screen-result/1",
        "status": (
            "FRESH_SCREEN_PASS_CONFIRMATION_COHORT_READY"
            if all(gates.values()) else "FRESH_SCREEN_FAIL"
        ),
        "screened": len(summaries), "active": len(active_rows),
        "active_scenes": len({row["scene_id"] for row in active_rows}),
        "selected_confirmation_cohort": selected,
        "gates": gates, "selection_used_task_metrics": False,
        "task_metric_payload_read": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "paper_result": False, "unseen_or_test_accessed": False,
    }
    atomic_json(RESULT, result)
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
