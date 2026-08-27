#!/usr/bin/env python3
"""Outcome-blind activation screen for the corrected R2R V5.3 overlay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/r2r_v5_3_activation_shadow_worker.py"
PILOT = ROOT / "scripts/r2r_action_enabled_pilot_worker_v5.py"
FUSION = ROOT / "revealnav_mf2r4/fusion.py"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr/val_seen/val_seen.json.gz"
)
CALIBRATION = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3/"
    "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_3_activation_screen"
PROTOCOL = OUT / "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_3_ACTIVATION_SCREEN_RESULT.json"
SALT = "revealnav-r2r-v5.3-outcome-blind-activation-screen/1"
SCREEN_SEED = 20260826
EPISODES_PER_SCENE = 2
ACTIVE_COHORT_LIMIT = 24


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


def selection() -> list[dict]:
    with gzip.open(DATASET, "rt") as stream:
        episodes = json.load(stream)["episodes"]
    grouped = defaultdict(list)
    for row in episodes:
        grouped[Path(row["scene_id"]).stem].append(row)
    if len(episodes) != 778 or len(grouped) != 53:
        raise RuntimeError("R2R val_seen inventory drift")
    selected = []
    for scene in sorted(grouped):
        ranked = sorted(grouped[scene], key=lambda row: hashlib.sha256(
            f"{SALT}|{scene}|{row['episode_id']}".encode()
        ).hexdigest())[:EPISODES_PER_SCENE]
        selected.extend({
            "episode_id": str(row["episode_id"]),
            "scene_id": scene,
            "trajectory_id": row.get("trajectory_id"),
            "screen_rank": hashlib.sha256(
                f"{SALT}|{scene}|{row['episode_id']}".encode()
            ).hexdigest(),
        } for row in ranked)
    return sorted(selected, key=lambda row: row["screen_rank"])


def protocol_value() -> dict:
    calibration = json.loads(CALIBRATION.read_text())
    config = calibration.get("selected_shared_config", {})
    if not (
        calibration.get("status") == "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_PASS"
        and calibration.get("gold_payload_read") is False
        and config.get("opv_threshold") == 0.025
    ):
        raise RuntimeError("frozen calibration precondition failed")
    rows = selection()
    return {
        "schema_version": "revealnav-r2r-v5.3-activation-screen-protocol/1",
        "status": "SEALED_BEFORE_OUTCOME_BLIND_ACTIVATION_SCREEN",
        "scope": "R2R val_seen internal development only",
        "selection_salt": SALT,
        "selection": rows,
        "screen_seed": SCREEN_SEED,
        "episodes_per_scene": EPISODES_PER_SCENE,
        "runs": len(rows),
        "selection_contract": {
            "worker_executes_no_controller_action": True,
            "worker_summary_contains_no_task_metrics": True,
            "verifier_must_not_open_etp_metric_files": True,
            "eligible": "activation_count > 0 under strict OPV > 0.025",
            "active_cohort_order": "predeclared screen_rank, not predicted gain",
            "active_cohort_limit": ACTIVE_COHORT_LIMIT,
            "navigation_outcomes_never_used_for_selection": True,
        },
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(PILOT.relative_to(ROOT)): sha256_file(PILOT),
            str(FUSION.relative_to(ROOT)): sha256_file(FUSION),
            str(DATASET.relative_to(ROOT)): sha256_file(DATASET),
            str(CALIBRATION.relative_to(ROOT)): sha256_file(CALIBRATION),
        },
        "test_or_test_challenge_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("activation screen protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "runs": value["runs"],
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def launch(episode: dict, gpu: int) -> dict:
    name = f"ep_{episode['episode_id']}"
    run_dir = OUT / "full/runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    logs = OUT / "full/logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"{name}.stdout.log").open("w")
    stderr = (logs / f"{name}.stderr.log").open("w")
    process = subprocess.Popen([
        str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
        "--episode-id", episode["episode_id"], "--run-dir", str(run_dir),
    ], cwd=ROOT, env={
        **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }, stdout=stdout, stderr=stderr)
    return {
        "name": name, "episode_id": episode["episode_id"], "gpu": gpu,
        "process": process, "streams": (stdout, stderr),
    }


def execute(gpus: tuple[int, ...], resume: bool) -> int:
    protocol = protocol_value()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("activation screen must be sealed")
    run_root = OUT / "full"
    if run_root.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite {run_root}")
    run_root.mkdir(parents=True, exist_ok=resume)
    completed = []
    completed_ids = set()
    if resume:
        for run_dir in sorted((run_root / "runs").glob("*")):
            path = run_dir / "RUN_SUMMARY.json"
            if path.is_file() and json.loads(path.read_text()).get("status") == "PASS":
                episode_id = str(json.loads(path.read_text())["episode_id"])
                completed_ids.add(episode_id)
                completed.append({
                    "name": run_dir.name, "episode_id": episode_id,
                    "returncode": 0, "recovered": True,
                })
            else:
                destination = run_root / "interrupted" / run_dir.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(run_dir, destination)
    queue = [row for row in protocol["selection"] if row["episode_id"] not in completed_ids]
    free = list(gpus)
    active = []
    while queue or active:
        while queue and free:
            active.append(launch(queue.pop(0), free.pop(0)))
        time.sleep(0.5)
        for item in list(active):
            code = item["process"].poll()
            if code is None:
                continue
            for stream in item["streams"]:
                stream.close()
            completed.append({
                "name": item["name"], "episode_id": item["episode_id"],
                "gpu": item["gpu"], "returncode": code,
            })
            active.remove(item)
            free.append(item["gpu"])
            free.sort()
    failures = [row for row in completed if row["returncode"] != 0]
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "PASS" if not failures else "FAIL",
        "completed": completed, "failures": failures,
    })
    return 0 if not failures else 1


def valid_chain(path: Path) -> bool:
    previous = "0" * 64
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        digest = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != claimed:
            return False
        previous = claimed
    return True


def verify() -> int:
    protocol = protocol_value()
    if json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("activation screen protocol drift")
    observed = {}
    for path in sorted((OUT / "full/runs").glob("*/RUN_SUMMARY.json")):
        row = json.loads(path.read_text())
        observed[str(row["episode_id"])] = (row, path.parent)
    expected = {row["episode_id"] for row in protocol["selection"]}
    gates = {
        "all_runs_complete": set(observed) == expected and all(
            row[0].get("status") == "PASS" for row in observed.values()
        ),
        "strict_checkpoints": all(
            row[0]["controller"]["strict_load"] for row in observed.values()
        ),
        "no_shadow_actions": all(
            row[0]["shadow_actions_executed"] == 0 for row in observed.values()
        ),
        "no_task_metric_payload_read": all(
            row[0]["task_metric_payload_read"] is False for row in observed.values()
        ),
        "opv_threshold_exact": all(
            row[0]["opv_threshold"] == 0.025 for row in observed.values()
        ),
        "valid_hash_chains": all(valid_chain(
            run_dir / "activation_trace.jsonl"
        ) for _, run_dir in observed.values()),
        "no_test_payload": True,
    }
    active = []
    for selected in protocol["selection"]:
        summary = observed[selected["episode_id"]][0]
        if summary["controller"]["activation_count"] > 0:
            active.append({
                **selected,
                "activation_count": summary["controller"]["activation_count"],
                "maximum_preservation_gain": summary["controller"]["maximum_preservation_gain"],
            })
    active_cohort = active[:ACTIVE_COHORT_LIMIT]
    result = {
        "schema_version": "revealnav-r2r-v5.3-activation-screen-result/1",
        "status": "ACTIVATION_SCREEN_PASS" if all(gates.values()) else "ACTIVATION_SCREEN_FAIL",
        "engineering_gates": gates,
        "screened_episodes": len(observed),
        "active_episodes": len(active),
        "active_scenes": len({row["scene_id"] for row in active}),
        "active_cohort": active_cohort,
        "active_cohort_limit": ACTIVE_COHORT_LIMIT,
        "selection_used_task_metrics": False,
        "result_contains_task_metrics": False,
        "protocol_sha256": sha256_file(PROTOCOL),
        "paper_result": False,
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain distinct device indices")
    if args.mode == "seal":
        return seal()
    if args.mode == "run":
        return execute(gpus, False)
    if args.mode == "resume":
        return execute(gpus, True)
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
