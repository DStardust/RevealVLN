#!/usr/bin/env python3
"""Fail-closed handoff from full training to R2R V5.13.1 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_r2r_v5_13_paired import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_TRAINING_RESULT,
    validate_training_result,
)
from run_r2r_v5_13_1_paired import paths as evaluation_paths  # noqa: E402


PYTHON = ROOT / ".envs/etpr1/bin/python"
RUNNER = ROOT / "scripts/run_r2r_v5_13_1_paired.py"
DATA_ROOT = ROOT / "artifacts/phase1/r2r_train_net_advantage"
DATA_PID = DATA_ROOT / "FULL_PIPELINE.pid"
DATA_PROGRESS = DATA_ROOT / "full/R2R_TRAIN_NET_ADVANTAGE_PROGRESS.json"
HANDOFF_ROOT = ROOT / "artifacts/evaluation/mf2_r2r_v5_14_net_advantage/handoff"
STATE = HANDOFF_ROOT / "HANDOFF_STATE.json"
PID = HANDOFF_ROOT / "HANDOFF.pid"
LOG = HANDOFF_ROOT / "HANDOFF.log"
DEFAULT_GPUS = "0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7"


class HandoffError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def write_state(stage: str, **values) -> None:
    atomic_json(STATE, {
        "schema_version": "revealnav-v5.13.1-automatic-handoff/1",
        "stage": stage,
        "updated_unix": time.time(),
        **values,
    })


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return False


def pid_from(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def wait_for_training(poll_seconds: int) -> dict:
    write_state(
        "WAITING_FULL_TRAINING", data_pipeline_pid=pid_from(DATA_PID),
        next="validate full three-seed training result",
    )
    while not DEFAULT_TRAINING_RESULT.is_file():
        data_pid = pid_from(DATA_PID)
        if not process_alive(data_pid):
            raise HandoffError(
                "BLOCKED_DATA_PIPELINE",
                "full data pipeline exited before writing the training result",
            )
        time.sleep(poll_seconds)
    try:
        result = validate_training_result(DEFAULT_TRAINING_RESULT)
    except Exception as error:
        raise HandoffError(
            "BLOCKED_TRAINING_GATE", f"{type(error).__name__}: {error}"
        ) from error
    write_state(
        "FULL_TRAINING_GATE_PASS", deployment=result["deployment"],
        next="launch complete val_seen paired matrix",
    )
    return result


def launch_or_attach(split: str, gpus: str, poll_seconds: int) -> dict:
    layout = evaluation_paths(split)
    result = load_json(layout["result"])
    if result is not None:
        return result
    evaluation_pid = pid_from(layout["pid"])
    if not process_alive(evaluation_pid):
        process = subprocess.run([
            str(PYTHON), str(RUNNER), "launch", "--split", split,
            "--gpus", gpus, "--protocol", str(DEFAULT_PROTOCOL),
            "--training-result", str(DEFAULT_TRAINING_RESULT),
        ], cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if process.returncode:
            raise HandoffError(
                f"BLOCKED_{split.upper()}_LAUNCH",
                process.stdout[-2000:],
            )
        evaluation_pid = pid_from(layout["pid"])
        if not process_alive(evaluation_pid):
            raise HandoffError(
                f"BLOCKED_{split.upper()}_LAUNCH",
                "evaluation launcher returned without a live orchestrator",
            )
    write_state(
        f"RUNNING_{split.upper()}", evaluation_pid=evaluation_pid,
        monitor=f"scripts/monitor_r2r_v5_13_1_paired.py --split {split}",
    )
    while True:
        result = load_json(layout["result"])
        if result is not None:
            return result
        if not process_alive(evaluation_pid):
            result = load_json(layout["result"])
            if result is not None:
                return result
            raise HandoffError(
                f"BLOCKED_{split.upper()}_EVALUATION",
                "evaluation orchestrator exited before writing its paired result",
            )
        time.sleep(poll_seconds)


def run(gpus: str, poll_seconds: int) -> int:
    try:
        wait_for_training(poll_seconds)
        seen = launch_or_attach("val_seen", gpus, poll_seconds)
        if seen.get("status") != "PASS":
            raise HandoffError(
                "BLOCKED_VAL_SEEN_SCIENTIFIC_GATE",
                "complete val_seen paired result did not pass the sealed main gates",
            )
        write_state(
            "VAL_SEEN_GATE_PASS", next="launch complete val_unseen paired matrix",
        )
        unseen = launch_or_attach("val_unseen", gpus, poll_seconds)
        write_state(
            "COMPLETE", val_seen_status=seen["status"],
            val_unseen_status=unseen.get("status"),
            paper_result=unseen.get("paper_result"),
        )
        return 0 if unseen.get("status") == "PASS" else 2
    except HandoffError as error:
        write_state(error.stage, error=str(error))
        raise
    except BaseException as error:
        write_state("FAILED_UNEXPECTED", error=f"{type(error).__name__}: {error}")
        raise


def launch(gpus: str, poll_seconds: int) -> int:
    current = pid_from(PID)
    if process_alive(current):
        raise RuntimeError(f"automatic handoff is already running as PID {current}")
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    log = LOG.open("a")
    process = subprocess.Popen([
        str(PYTHON), str(Path(__file__).resolve()), "run",
        "--gpus", gpus, "--poll-seconds", str(poll_seconds),
    ], cwd=ROOT, env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    PID.write_text(f"{process.pid}\n")
    return process.pid


def monitor() -> dict:
    data = load_json(DATA_PROGRESS) or {}
    training = load_json(DEFAULT_TRAINING_RESULT) or {}
    evaluations = {}
    for split in ("val_seen", "val_unseen"):
        layout = evaluation_paths(split)
        status = load_json(layout["status"]) or {}
        result = load_json(layout["result"]) or {}
        evaluations[split] = {
            "orchestrator_pid": pid_from(layout["pid"]),
            "orchestrator_alive": process_alive(pid_from(layout["pid"])),
            "completed": status.get("completed"),
            "expected": status.get("expected"),
            "active": status.get("active", []),
            "failures": len(status.get("failures", [])),
            "result_status": result.get("status"),
        }
    supervisor_pid = pid_from(PID)
    return {
        "supervisor_pid": supervisor_pid,
        "supervisor_alive": process_alive(supervisor_pid),
        "handoff": load_json(STATE),
        "data_generation": {
            "status": data.get("status"), "completed": data.get("completed"),
            "selected": data.get("selected"), "remaining": data.get("remaining"),
            "feature_events": data.get("feature_events"),
            "failures": len(data.get("failures", [])),
        },
        "training_status": training.get("status"),
        "evaluations": evaluations,
        "log": str(LOG.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "run", "monitor"))
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise SystemExit("poll interval must be at least five seconds")
    if args.command == "launch":
        supervisor_pid = launch(args.gpus, args.poll_seconds)
        print(json.dumps({
            "status": "LAUNCHED", "pid": supervisor_pid,
            "monitor": "scripts/watch_r2r_v5_13_1_handoff.py monitor",
        }, sort_keys=True))
        return 0
    if args.command == "run":
        return run(args.gpus, args.poll_seconds)
    print(json.dumps(monitor(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
