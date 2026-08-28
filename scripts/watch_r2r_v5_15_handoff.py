#!/usr/bin/env python3
"""Persistently hand V5.15 train-only calibration to val-seen evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
RUNNER = ROOT / "scripts/run_r2r_v5_15_paired.py"
CAL_PROGRESS = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/"
    "R2R_TRAIN_V5_15_POLICY_PROGRESS.json"
)
CAL_RESULT = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/calibration/"
    "R2R_V5_15_POLICY_CALIBRATION_RESULT.json"
)
ROOT_OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_15_policy_calibrated/handoff"
STATE = ROOT_OUT / "HANDOFF_STATE.json"
PID = ROOT_OUT / "HANDOFF.pid"
LOG = ROOT_OUT / "HANDOFF.log"
EVAL_PID = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_15_policy_calibrated/val_seen/"
    "ORCHESTRATOR.pid"
)
EVAL_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_15_policy_calibrated/val_seen/"
    "R2R_V5_15_PAIRED_RESULT.json"
)
DEFAULT_GPUS = "0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7"


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def process_alive(path: Path) -> bool:
    try:
        os.kill(int(path.read_text()), 0)
        return True
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        return False


def write_state(stage: str, **values) -> None:
    atomic_json(STATE, {
        "schema_version": "revealnav-r2r-v5.15-handoff/1",
        "stage": stage, "updated_unix": time.time(),
        "val_unseen_or_test_read": False, **values,
    })


def run(gpus: str, poll_seconds: int) -> int:
    while True:
        progress = load(CAL_PROGRESS)
        write_state(
            "WAITING_POLICY_CALIBRATION",
            calibration_status=progress.get("status"),
            calibration_stage=progress.get("stage"),
            completed=progress.get("completed"), selected=progress.get("selected"),
            proposal_events=progress.get("proposal_events"),
        )
        if progress.get("status") in ("CALIBRATION_PASS", "CALIBRATION_FAIL"):
            break
        time.sleep(poll_seconds)
    if progress["status"] != "CALIBRATION_PASS":
        write_state(
            "BLOCKED_CALIBRATION_GATE",
            calibration_result=load(CAL_RESULT),
        )
        return 2
    process = subprocess.run(
        [str(PYTHON), str(RUNNER), "launch", "--gpus", gpus],
        cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if process.returncode:
        write_state("BLOCKED_EVALUATION_LAUNCH", error=process.stdout[-3000:])
        return 2
    write_state("RUNNING_VAL_SEEN", launch=process.stdout.strip())
    while not EVAL_RESULT.is_file():
        if not process_alive(EVAL_PID):
            write_state("BLOCKED_EVALUATION_EXITED_WITHOUT_RESULT")
            return 2
        time.sleep(poll_seconds)
    result = load(EVAL_RESULT)
    write_state(
        "COMPLETE", result_status=result.get("status"),
        main_gates=result.get("main_gates"), paper_result=False,
    )
    return 0 if result.get("status") == "PASS" else 2


def launch(gpus: str, poll_seconds: int) -> int:
    if process_alive(PID):
        raise RuntimeError("V5.15 handoff is already running")
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    stream = LOG.open("a")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "run", "--gpus", gpus,
         "--poll-seconds", str(poll_seconds)],
        cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    )
    stream.close()
    PID.write_text(f"{process.pid}\n")
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "run", "monitor"))
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.command == "launch":
        print(json.dumps({"status": "LAUNCHED", "pid": launch(args.gpus, args.poll_seconds)}))
        return 0
    if args.command == "run":
        return run(args.gpus, args.poll_seconds)
    progress = load(CAL_PROGRESS)
    evaluation_process = subprocess.run(
        [str(PYTHON), str(RUNNER), "monitor"], cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        evaluation = json.loads(evaluation_process.stdout)
    except json.JSONDecodeError:
        evaluation = {"monitor_error": evaluation_process.stdout[-1000:]}
    completed = int(progress.get("completed", 0))
    selected = int(progress.get("selected", 0))
    print(json.dumps({
        "supervisor_alive": process_alive(PID),
        "state": load(STATE),
        "calibration": {
            "status": progress.get("status"),
            "stage": progress.get("stage"),
            "completed": completed,
            "selected": selected,
            "progress_percent": (
                round(100 * completed / selected, 2) if selected else 0
            ),
            "active": len(progress.get("active", [])),
            "proposal_events": progress.get("proposal_events"),
            "proposal_events_by_seed": progress.get("proposal_events_by_seed"),
            "missing_causal_inputs": progress.get("missing_causal_inputs"),
            "exhausted_failures": len(progress.get("exhausted_failures", [])),
        },
        "evaluation": evaluation,
        "log": str(LOG.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
