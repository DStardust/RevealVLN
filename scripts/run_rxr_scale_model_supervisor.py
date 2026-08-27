#!/usr/bin/env python3
"""Wait for automatic scale features, then run the sealed model comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2_STATUS = BASE / "scale_v2/RXR_SCALE_V2_SUPERVISOR_STATUS.json"
MODEL_ROOT = BASE / "scale_v2/model_training"
STATUS = MODEL_ROOT / "RXR_SCALE_MODEL_SUPERVISOR_STATUS.json"
LOG = MODEL_ROOT / "RXR_SCALE_MODEL_SUPERVISOR.log"
FEATURE_GATES = (
    BASE / "scale_v1/automatic/multibranch/RXR_SCALE_FEATURE_GATE.json",
    BASE / "scale_v2/automatic/multibranch/RXR_SCALE_FEATURE_GATE.json",
)
TRAINER = ROOT / "scripts/run_rxr_scale_relational_training.py"
SEEDS = (20260826, 20260827, 20260828)


def atomic_status(value: dict) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def feature_ready() -> bool:
    return all(
        path.is_file()
        and json.loads(path.read_text()).get("status")
        == "FEATURE_GATE_PASS_AUTOMATIC_SCALE_READY"
        for path in FEATURE_GATES
    )


def wait_for_scale(state: dict) -> None:
    while True:
        if V2_STATUS.is_file():
            value = json.loads(V2_STATUS.read_text())
            completed = value.get("status") == "SCALE_V2_EVENT_EXPANSION_PASS_GOLD_REVIEWS_REQUIRED"
            expected_capacity_stop = (
                value.get("status") == "FAIL"
                and value.get("failed_stage") == "finalize_capacity"
            )
            if (completed or expected_capacity_stop) and feature_ready():
                state["scale_release"] = "COMPLETE" if completed else "EXPECTED_CAPACITY_SHORTFALL"
                return
            if value.get("status") == "FAIL" and not expected_capacity_stop:
                raise RuntimeError(f"scale-v2 failed before feature closure: {value.get('failed_stage')}")
        state.update({"status": "WAITING_FOR_AUTOMATIC_SCALE_FEATURES", "updated": time.time()})
        atomic_status(state)
        time.sleep(20)


def run_stage(state: dict, name: str, command: list[str], log) -> None:
    row = {"name": name, "status": "RUNNING", "started": time.time()}
    state.update({"status": "RUNNING", "current_stage": name})
    state["stages"].append(row); atomic_status(state)
    result = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "HOME": str(ROOT)},
        stdout=log, stderr=subprocess.STDOUT, check=False,
    )
    row.update({
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - row["started"], 3),
    })
    atomic_status(state)
    if result.returncode:
        raise RuntimeError(f"stage failed: {name}")


def train_seeds(state: dict) -> None:
    row = {"name": "three_seed_matched_training", "status": "RUNNING", "started": time.time()}
    state.update({"status": "RUNNING", "current_stage": row["name"]})
    state["stages"].append(row); atomic_status(state)
    processes = []
    handles = []
    try:
        for gpu, seed in enumerate(SEEDS):
            path = MODEL_ROOT / f"seed_{seed}.log"
            handle = path.open("a")
            handles.append(handle)
            processes.append((seed, subprocess.Popen(
                [sys.executable, str(TRAINER), "--seed", str(seed), "--device", "cuda"],
                cwd=ROOT,
                env={**os.environ, "HOME": str(ROOT), "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=handle, stderr=subprocess.STDOUT,
            )))
        completed = [(seed, process.wait()) for seed, process in processes]
    except Exception:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()
    failures = [(seed, code) for seed, code in completed if code]
    row.update({
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "elapsed_seconds": round(time.time() - row["started"], 3),
    })
    atomic_status(state)
    if failures:
        raise RuntimeError(f"scale model seed failures: {failures}")


def main() -> int:
    state = {
        "schema_version": "revealnav-scale-model-supervisor/1",
        "status": "STARTING", "pid": os.getpid(), "started": time.time(),
        "stages": [],
    }
    atomic_status(state)
    try:
        wait_for_scale(state)
        with LOG.open("a") as log:
            run_stage(state, "build_training_manifest", [
                sys.executable, "scripts/build_rxr_scale_automatic_training_manifest.py",
            ], log)
            run_stage(state, "seal_training_protocol", [
                sys.executable, str(TRAINER), "--seal",
            ], log)
            train_seeds(state)
            run_stage(state, "aggregate_model_comparison", [
                sys.executable, str(TRAINER), "--aggregate",
            ], log)
        result = json.loads((ROOT / "artifacts/evaluation/mf2_scale_relational_v1/RXR_SCALE_RELATIONAL_COMPARISON_V1.json").read_text())
        state.pop("current_stage", None)
        state.update({
            "status": "SCALE_MODEL_COMPARISON_COMPLETE",
            "model_gate": result["status"],
            "selected_model": result["selected_model"],
            "completed_at": time.time(),
        })
        atomic_status(state)
        return 0
    except Exception as error:
        state.update({"status": "FAIL", "error": repr(error), "failed_at": time.time()})
        atomic_status(state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
