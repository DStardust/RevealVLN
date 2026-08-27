#!/usr/bin/env python3
"""Wait for the data ablation, then run relational augmented training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASELINE_COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_secondary_augmentation_v1/"
    "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_relational_augmented_v2"
STATUS = OUT / "RXR_RELATIONAL_AUGMENTED_STATUS_V2.json"
CONSOLE = OUT / "RXR_RELATIONAL_AUGMENTED_SUPERVISOR.log"
COMPARISON = OUT / "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
SEEDS = (20260826, 20260827, 20260828)


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def complete(seed: int) -> bool:
    path = OUT / f"seed_{seed}/result.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text())
    return value.get("status") == "RELATIONAL_AUGMENTED_RUN_COMPLETE"


def supervise() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "revealnav-mf2-relational-augmented-status/2",
        "status": "WAITING_FOR_FROZEN_AUGMENTATION_ABLATION",
        "pid": os.getpid(),
        "started": time.time(),
        "runs": {},
    }
    atomic_json(STATUS, state)
    while not BASELINE_COMPARISON.is_file():
        state["last_wait_update"] = time.time()
        atomic_json(STATUS, state)
        time.sleep(15)

    state["status"] = "TRAINING_RELATIONAL_AUGMENTED"
    pending = [seed for seed in SEEDS if not complete(seed)]
    running: dict[int, tuple[subprocess.Popen, object, int]] = {}
    available_gpus = [0, 1]
    while pending or running:
        while pending and available_gpus:
            seed = pending.pop(0)
            gpu = available_gpus.pop(0)
            run_dir = OUT / f"seed_{seed}"
            if run_dir.exists():
                raise RuntimeError(f"incomplete run directory: {run_dir}")
            log = (OUT / f"seed_{seed}.log").open("a")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/run_rxr_relational_augmented_v2.py",
                    "--seed", str(seed), "--device", "cuda",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[seed] = (process, log, gpu)
            state["runs"][str(seed)] = {
                "status": "RUNNING", "physical_gpu": gpu,
                "pid": process.pid,
            }
        atomic_json(STATUS, state)
        for seed, (process, log, gpu) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[seed]
            available_gpus.append(gpu)
            available_gpus.sort()
            state["runs"][str(seed)].update({
                "status": "PASS" if code == 0 else "FAIL",
                "returncode": code,
            })
            if code:
                state["status"] = "FAIL_RELATIONAL_AUGMENTED_RUN"
                atomic_json(STATUS, state)
                return code
        if pending or running:
            time.sleep(5)
    if not all(complete(seed) for seed in SEEDS):
        raise RuntimeError("relational augmented run closure failed")

    state["status"] = "AGGREGATING_2X2_ABLATION"
    atomic_json(STATUS, state)
    with (OUT / "aggregate.log").open("a") as log:
        code = subprocess.run(
            [
                sys.executable,
                "scripts/run_rxr_relational_augmented_v2.py", "--aggregate",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if code or not COMPARISON.is_file():
        state["status"] = "FAIL_2X2_AGGREGATION"
        state["aggregate_returncode"] = code
        atomic_json(STATUS, state)
        return code or 1
    comparison = json.loads(COMPARISON.read_text())
    state.update({
        "status": "RELATIONAL_AUGMENTED_ABLATION_COMPLETE",
        "completed": time.time(),
        "result_status": comparison["status"],
        "selected_training_condition": comparison["selected_training_condition"],
    })
    atomic_json(STATUS, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        OUT.mkdir(parents=True, exist_ok=True)
        with CONSOLE.open("a") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"status": "DETACHED", "pid": process.pid}, indent=2))
        return 0
    return supervise()


if __name__ == "__main__":
    raise SystemExit(main())
