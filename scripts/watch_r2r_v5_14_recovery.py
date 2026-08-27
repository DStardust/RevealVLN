#!/usr/bin/env python3
"""Operationally recover failed paired jobs without changing sealed evaluation code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_r2r_v5_13_1_paired import paths  # noqa: E402
from watch_r2r_v5_13_1_handoff import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_TRAINING_RESULT,
    PID as HANDOFF_PID,
    process_alive,
)


PYTHON = ROOT / ".envs/etpr1/bin/python"
RUNNER = SCRIPTS / "run_r2r_v5_13_1_paired.py"
HANDOFF = SCRIPTS / "watch_r2r_v5_13_1_handoff.py"
ROOT_OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_14_net_advantage/recovery"
PID = ROOT_OUT / "RECOVERY.pid"
STATE = ROOT_OUT / "RECOVERY_STATE.json"
LOG = ROOT_OUT / "RECOVERY.log"
# GPUs 2--7 carry company placeholder workloads, so they receive one job each.
SAFE_GPUS = "0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,3,4,5,6,7"


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


def write_state(stage: str, **values: object) -> None:
    atomic_json(STATE, {
        "schema_version": "revealnav-v5.14-operational-recovery/1",
        "stage": stage,
        "updated_unix": time.time(),
        **values,
    })


def pid(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def command(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PYTHON), *arguments], cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"}, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def recover(split: str, poll_seconds: int) -> dict:
    layout = paths(split)
    attempt = 0
    while not layout["result"].is_file():
        attempt += 1
        write_state(
            f"RECOVERING_{split.upper()}", attempt=attempt,
            safe_gpus=SAFE_GPUS,
        )
        result = command([
            str(RUNNER), "resume", "--split", split, "--gpus", SAFE_GPUS,
            "--protocol", str(DEFAULT_PROTOCOL),
            "--training-result", str(DEFAULT_TRAINING_RESULT),
        ])
        with LOG.open("a") as stream:
            stream.write(result.stdout)
        if result.returncode:
            write_state(
                f"RETRYING_{split.upper()}", attempt=attempt,
                returncode=result.returncode,
            )
            time.sleep(poll_seconds)
            continue
        verified = command([
            str(RUNNER), "verify", "--split", split,
            "--protocol", str(DEFAULT_PROTOCOL),
            "--training-result", str(DEFAULT_TRAINING_RESULT),
        ])
        with LOG.open("a") as stream:
            stream.write(verified.stdout)
        if not layout["result"].is_file():
            write_state(
                f"RETRYING_{split.upper()}_VERIFY",
                returncode=verified.returncode,
            )
            time.sleep(poll_seconds)
    return load(layout["result"])


def run(poll_seconds: int) -> int:
    write_state("ATTACHED", safe_gpus=SAFE_GPUS)
    while True:
        handoff_pid = pid(HANDOFF_PID)
        if process_alive(handoff_pid):
            time.sleep(poll_seconds)
            continue
        seen = load(paths("val_seen")["result"])
        if not seen:
            seen_pid = pid(paths("val_seen")["pid"])
            if process_alive(seen_pid):
                time.sleep(poll_seconds)
                continue
            seen = recover("val_seen", poll_seconds)
        if seen.get("status") != "PASS":
            write_state("BLOCKED_VAL_SEEN_SCIENTIFIC_GATE", status=seen.get("status"))
            return 2
        unseen = load(paths("val_unseen")["result"])
        if not unseen:
            unseen_pid = pid(paths("val_unseen")["pid"])
            if process_alive(unseen_pid):
                time.sleep(poll_seconds)
                continue
            status = load(paths("val_unseen")["status"])
            if status:
                unseen = recover("val_unseen", poll_seconds)
            else:
                launched = command([
                    str(HANDOFF), "launch", "--gpus", SAFE_GPUS,
                    "--poll-seconds", str(poll_seconds),
                ])
                with LOG.open("a") as stream:
                    stream.write(launched.stdout)
                if launched.returncode:
                    write_state("RETRYING_HANDOFF_LAUNCH", returncode=launched.returncode)
                    time.sleep(poll_seconds)
                continue
        write_state(
            "COMPLETE" if unseen.get("status") == "PASS" else
            "COMPLETE_WITH_NEGATIVE_SCIENTIFIC_RESULT",
            val_seen_status=seen.get("status"),
            val_unseen_status=unseen.get("status"),
        )
        return 0 if unseen.get("status") == "PASS" else 2


def launch(poll_seconds: int) -> int:
    current = pid(PID)
    if process_alive(current):
        raise RuntimeError(f"recovery supervisor already running as PID {current}")
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    stream = LOG.open("a")
    process = subprocess.Popen(
        [str(PYTHON), str(Path(__file__).resolve()), "run",
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
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise SystemExit("poll interval must be at least five seconds")
    if args.command == "launch":
        print(json.dumps({"status": "LAUNCHED", "pid": launch(args.poll_seconds)}))
        return 0
    if args.command == "run":
        return run(args.poll_seconds)
    current = pid(PID)
    print(json.dumps({
        "pid": current, "alive": process_alive(current), "state": load(STATE),
        "log": str(LOG.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
