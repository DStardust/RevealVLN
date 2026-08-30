#!/usr/bin/env python3
"""Continue from V6 pilot feasibility to full train-only cross-fitting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
BASE = ROOT / "artifacts/phase1/rxr_v6"
STATE = BASE / "RXR_V6_CAMPAIGN_STATE.json"


def atomic_json(value: dict) -> None:
    part = STATE.with_name(STATE.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATE)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(command: list[str], log: Path, env_update: dict | None = None) -> int:
    env = dict(os.environ)
    if env_update:
        env.update(env_update)
    with log.open("w") as stream:
        return subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            check=False, text=True, env=env,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-pid", type=int, required=True)
    args = parser.parse_args()
    pilot_manifest = BASE / "pilot_v6_0/RXR_V6_PAIRED_DATASET_MANIFEST.json"
    while process_alive(args.pilot_pid):
        atomic_json({
            "status": "WAITING_FOR_PILOT", "pilot_pid": args.pilot_pid,
        })
        time.sleep(60)
    if not pilot_manifest.is_file():
        atomic_json({
            "status": "PILOT_EXECUTION_FAILED", "pilot_pid": args.pilot_pid,
        })
        return 2
    pilot = json.loads(pilot_manifest.read_text())["metadata"]
    pilot_gates = {
        "at_least_ten_pairs": pilot["pairs"] >= 10,
        "at_least_two_positive_pairs": pilot["positive_pairs"] >= 2,
        "positive_rate_at_least_five_percent": (
            pilot["positive_pairs"] / pilot["pairs"] >= 0.05
        ),
    }
    if not all(pilot_gates.values()):
        atomic_json({
            "status": "PILOT_FEASIBILITY_FAIL",
            "pilot": pilot, "pilot_gates": pilot_gates,
            "full_collection_started": False,
        })
        return 2
    atomic_json({
        "status": "FULL_COLLECTION_RUNNING", "pilot": pilot,
        "pilot_gates": pilot_gates, "full_collection_started": True,
    })
    full_log = BASE / "full_v6_0/PIPELINE.log"
    full_log.parent.mkdir(parents=True, exist_ok=True)
    full_return = run([
        str(PYTHON), "scripts/run_rxr_v6_counterfactual_pipeline.py", "all",
        "--cohort", "full_v6_0", "--episodes", "480",
        "--gpus", "2,3,4,5,6,7",
    ], full_log)
    full_manifest = BASE / "full_v6_0/RXR_V6_PAIRED_DATASET_MANIFEST.json"
    if full_return or not full_manifest.is_file():
        atomic_json({
            "status": "FULL_COLLECTION_FAILED", "pilot": pilot,
            "pilot_gates": pilot_gates, "returncode": full_return,
        })
        return 2
    atomic_json({
        "status": "OFFLINE_CROSSFIT_RUNNING", "pilot": pilot,
        "pilot_gates": pilot_gates,
        "full": json.loads(full_manifest.read_text())["metadata"],
    })
    output = BASE / "full_v6_0/training"
    output.mkdir(parents=True, exist_ok=True)
    train_return = run([
        str(PYTHON), "scripts/train_rxr_v6_relative_advantage.py",
        "--manifest", str(full_manifest), "--output-dir", str(output),
        "--device", "cuda:0",
    ], BASE / "full_v6_0/TRAINING.log", {
        "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
    })
    result = output / "RXR_V6_OFFLINE_CROSSFIT_RESULT.json"
    value = json.loads(result.read_text()) if result.is_file() else None
    atomic_json({
        "status": (
            "V6_OFFLINE_GATE_PASS" if train_return == 0
            else "V6_OFFLINE_GATE_FAIL"
        ),
        "pilot": pilot, "pilot_gates": pilot_gates,
        "full": json.loads(full_manifest.read_text())["metadata"],
        "training_returncode": train_return,
        "offline_result": value,
    })
    return train_return


if __name__ == "__main__":
    raise SystemExit(main())
