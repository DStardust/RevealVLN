#!/usr/bin/env python3
"""Run the secondary augmentation pipeline after multiview rendering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
MULTIVIEW = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
STATUS = BASE / "RXR_SECONDARY_AUTOMATIC_PIPELINE_STATUS.json"
LOG = BASE / "RXR_SECONDARY_AUTOMATIC_PIPELINE.log"


STAGES = (
    ("branch_first_response", [
        sys.executable, "scripts/run_rxr_secondary_branch_stage.py"
    ], {}),
    ("multibranch_geometry", [
        sys.executable, "scripts/run_rxr_secondary_multibranch_geometry.py"
    ], {}),
    ("multibranch_controller", [
        sys.executable, "scripts/run_rxr_secondary_multibranch_controller.py"
    ], {"CUDA_VISIBLE_DEVICES": "0", "CR5_CONTROLLER_GPU": "0"}),
    ("causal_frontend", [
        sys.executable, "scripts/run_rxr_secondary_frontend_stage.py"
    ], {}),
    ("causal_analysis", [
        sys.executable, "scripts/analyze_rxr_secondary_multibranch_causal.py"
    ], {}),
    ("causal_media", [
        sys.executable, "scripts/build_rxr_secondary_causal_prefix_media.py"
    ], {"CUDA_VISIBLE_DEVICES": "0", "CR5_CAUSAL_MEDIA_GPU": "0"}),
    ("causal_language", [
        sys.executable, "scripts/run_rxr_secondary_causal_prefix_language.py",
        "--execute", "--workers", "16",
    ], {}),
    ("training_index", [
        sys.executable, "scripts/build_rxr_secondary_training_index.py"
    ], {}),
    ("resource_labels", [
        sys.executable, "scripts/run_rxr_secondary_tx.py",
        "--gpus", "0,1,2,3,4,5,6,7",
    ], {}),
    ("frozen_features", [
        sys.executable, "scripts/run_rxr_secondary_features.py",
        "--gpus", "0,1,2,3,4,5,6,7",
    ], {}),
)


def atomic_status(value: dict) -> None:
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def wait_for_multiview() -> dict:
    while True:
        if MULTIVIEW.is_file() and not MULTIVIEW.is_symlink():
            value = json.loads(MULTIVIEW.read_text())
            if value.get("status") == "READY_FOR_BRANCH_PROPOSER":
                return value
        time.sleep(10)


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "WAITING_FOR_MULTIVIEW",
        "pid": os.getpid(),
        "stages": [],
        "training_authorized": False,
    }
    atomic_status(state)
    manifest = wait_for_multiview()
    state["multiview"] = {
        "event_count": manifest["event_count"],
        "failure_count": manifest["failure_count"],
    }
    with LOG.open("a") as log:
        for name, command, additions in STAGES:
            started = time.time()
            stage = {"name": name, "status": "RUNNING", "started": started}
            state["status"] = "RUNNING"
            state["current_stage"] = name
            state["stages"].append(stage)
            atomic_status(state)
            environment = {**os.environ, **additions}
            log.write(f"\n[{name}] START {started}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            stage.update({
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "returncode": result.returncode,
                "elapsed_seconds": round(time.time() - started, 3),
            })
            atomic_status(state)
            if result.returncode != 0:
                state["status"] = "FAIL"
                atomic_status(state)
                return result.returncode
        state["status"] = "AUTOMATIC_PIPELINE_PASS_TRAIN_MERGE_READY"
        state.pop("current_stage", None)
        atomic_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
