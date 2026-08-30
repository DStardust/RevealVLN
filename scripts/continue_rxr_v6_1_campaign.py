#!/usr/bin/env python3
"""Guarded V6.x pilot-to-full continuation with a 30-minute shadow cap."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
BASE = ROOT / "artifacts/phase1/rxr_v6"
STATE = BASE / "RXR_V6_1_CAMPAIGN_STATE.json"


def atomic_json(value: dict) -> None:
    part = STATE.with_name(STATE.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATE)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def execute(command: list[str], log: Path, env_update=None) -> int:
    env = dict(os.environ)
    if env_update:
        env.update(env_update)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(
            command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True, check=False,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-pid", required=True, type=int)
    parser.add_argument("--revision", choices=("v6_1", "v6_2"), default="v6_1")
    args = parser.parse_args()
    global STATE
    label = args.revision.upper()
    pilot_cohort = f"pilot_{args.revision}"
    full_cohort = f"full_{args.revision}"
    runner = f"scripts/run_rxr_{args.revision}_counterfactual_pipeline.py"
    STATE = BASE / f"RXR_{label}_CAMPAIGN_STATE.json"
    pilot_manifest = BASE / pilot_cohort / "RXR_V6_PAIRED_DATASET_MANIFEST.json"
    while alive(args.pilot_pid):
        atomic_json({
            "status": f"WAITING_FOR_{label}_PILOT", "pilot_pid": args.pilot_pid,
        })
        time.sleep(30)
    if not pilot_manifest.is_file():
        atomic_json({"status": f"{label}_PILOT_EXECUTION_FAILED"})
        return 2
    pilot = json.loads(pilot_manifest.read_text())["metadata"]
    gates = {
        "at_least_twelve_pairs": pilot["pairs"] >= 12,
        "at_least_two_positive_pairs": pilot["positive_pairs"] >= 2,
        "positive_rate_at_least_five_percent": (
            pilot["positive_pairs"] / pilot["pairs"] >= 0.05
        ),
    }
    if not all(gates.values()):
        atomic_json({
            "status": f"{label}_PILOT_FEASIBILITY_FAIL",
            "pilot": pilot, "gates": gates,
        })
        return 2
    rate = pilot["pairs"] / 30.0
    full_episodes = min(180, max(120, math.ceil(150.0 / rate)))
    atomic_json({
        "status": f"{label}_FULL_COLLECTION_RUNNING", "pilot": pilot,
        "gates": gates, "full_episodes": full_episodes,
        "shadow_wall_clock_contract_minutes": 30,
    })
    full_root = BASE / full_cohort
    code = execute([
        str(PYTHON), runner, "all",
        "--cohort", full_cohort, "--episodes", str(full_episodes),
        "--gpus", "0,0,0,0,1,1,1,1,2,3,4,5,6,7",
    ], full_root / "PIPELINE.log")
    manifest = full_root / "RXR_V6_PAIRED_DATASET_MANIFEST.json"
    if code or not manifest.is_file():
        atomic_json({
            "status": f"{label}_FULL_COLLECTION_FAILED", "pilot": pilot,
            "gates": gates, "full_episodes": full_episodes,
            "returncode": code,
        })
        return 2
    full = json.loads(manifest.read_text())["metadata"]
    atomic_json({
        "status": f"{label}_CROSSFIT_RUNNING", "pilot": pilot,
        "gates": gates, "full_episodes": full_episodes, "full": full,
    })
    output = full_root / "training"
    code = execute([
        str(PYTHON), "scripts/train_rxr_v6_relative_advantage.py",
        "--manifest", str(manifest), "--output-dir", str(output),
        "--device", "cuda:0",
    ], full_root / "TRAINING.log", {
        "CUDA_VISIBLE_DEVICES": "2", "PYTHONNOUSERSITE": "1",
    })
    result = output / "RXR_V6_OFFLINE_CROSSFIT_RESULT.json"
    atomic_json({
        "status": (
            f"{label}_OFFLINE_GATE_PASS" if code == 0
            else f"{label}_OFFLINE_GATE_FAIL"
        ),
        "pilot": pilot, "gates": gates,
        "full_episodes": full_episodes, "full": full,
        "offline_result": (
            json.loads(result.read_text()) if result.is_file() else None
        ),
    })
    return code


if __name__ == "__main__":
    raise SystemExit(main())
