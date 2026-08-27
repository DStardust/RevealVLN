#!/usr/bin/env python3
"""Durably supervise scale-v1 branch, automatic closure, and Gold packaging."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
MULTIVIEW_STATUS = BASE / "RXR_SCALE_V1_MULTIVIEW_STATUS.json"
STATUS = BASE / "RXR_SCALE_V1_SUPERVISOR_STATUS.json"
LOG = BASE / "RXR_SCALE_V1_SUPERVISOR.log"
CAUSAL_WORKERS = os.environ.get("RXR_SCALE_CAUSAL_WORKERS", "48")

STAGES = (
    ("build_gold_review_package", [sys.executable, "scripts/build_rxr_new_gold_review_package.py"], {}),
    ("validate_gold_review_package", [sys.executable, "scripts/validate_rxr_new_gold_review_package.py"], {}),
    ("branch_first_responses", [sys.executable, "scripts/run_rxr_scale_v1_branch_stage.py"], {}),
    ("automatic_geometry", [sys.executable, "scripts/run_rxr_scale_v1_geometry.py"], {}),
    ("automatic_controller", [sys.executable, "scripts/run_rxr_scale_v1_controller.py"], {"CUDA_VISIBLE_DEVICES": "0", "CR5_CONTROLLER_GPU": "0"}),
    ("automatic_frontend", [sys.executable, "scripts/run_rxr_scale_v1_frontend_stage.py"], {}),
    ("automatic_causal_analysis", [sys.executable, "scripts/analyze_rxr_scale_v1_causal.py"], {}),
    ("automatic_causal_media", [sys.executable, "scripts/build_rxr_scale_v1_causal_media.py"], {"CUDA_VISIBLE_DEVICES": "0", "CR5_CAUSAL_MEDIA_GPU": "0"}),
    ("automatic_causal_language", [sys.executable, "scripts/run_rxr_scale_v1_causal_language.py", "--execute", "--workers", CAUSAL_WORKERS], {}),
    ("automatic_training_index", [sys.executable, "scripts/build_rxr_scale_v1_training_index.py"], {}),
    ("automatic_resource_labels", [sys.executable, "scripts/run_rxr_scale_v1_tx.py", "--gpus", "0,1,2,3,4,5,6,7"], {}),
    ("automatic_frozen_features", [sys.executable, "scripts/run_rxr_scale_v1_features.py", "--gpus", "0,1,2,3,4,5,6,7"], {}),
    ("finalize_scale_capacity", [sys.executable, "scripts/finalize_rxr_scale_v1_expansion.py"], {}),
)


def atomic_status(value: dict) -> None:
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def wait_for_multiview(state: dict) -> None:
    while True:
        if MULTIVIEW_STATUS.is_file():
            value = json.loads(MULTIVIEW_STATUS.read_text())
            if value.get("status") == "SCALE_V1_MULTIVIEW_PASS":
                state["multiview"] = value["status"]
                atomic_status(state)
                return
            if value.get("status") == "FAIL":
                raise RuntimeError("scale multiview failed")
        state["status"] = "WAITING_FOR_MULTIVIEW"
        state["updated"] = time.time()
        atomic_status(state)
        time.sleep(20)


def main() -> int:
    resume_stage = os.environ.get("RXR_SCALE_V1_RESUME_STAGE")
    if resume_stage:
        names = [row[0] for row in STAGES]
        if resume_stage not in names or not STATUS.is_file():
            raise RuntimeError("invalid scale-v1 resume stage")
        state = json.loads(STATUS.read_text())
        failed_resume = (
            state.get("status") == "FAIL"
            and state.get("failed_stage") == resume_stage
        )
        planned_resume = (
            os.environ.get("RXR_SCALE_V1_ALLOW_PLANNED_RESUME") == "1"
            and state.get("status") == "RUNNING"
            and state.get("current_stage") == resume_stage
        )
        if not (failed_resume or planned_resume):
            raise RuntimeError("resume stage does not match recorded state")
        if planned_resume:
            running = state.get("stages", [])[-1]
            if running.get("name") != resume_stage or running.get("status") != "RUNNING":
                raise RuntimeError("planned resume has no matching running stage")
            running.update({
                "status": "INTERRUPTED_FOR_CONCURRENCY_CHANGE",
                "interrupted_at": time.time(),
            })
        stages = STAGES[names.index(resume_stage):]
        state.update({
            "status": "RESUMING",
            "pid": os.getpid(),
            "resumed_at": time.time(),
            "resume_stage": resume_stage,
            "causal_language_workers": int(CAUSAL_WORKERS),
        })
        state.pop("failed_stage", None)
        state.pop("error", None)
    else:
        state = {
            "schema_version": "revealnav-rxr-scale-supervisor-status/1",
            "status": "STARTING",
            "pid": os.getpid(),
            "started": time.time(),
            "stages": [],
        }
        stages = STAGES
    atomic_status(state)
    try:
        wait_for_multiview(state)
    except Exception as error:
        state.update({"status": "FAIL", "error": repr(error)})
        atomic_status(state)
        return 1
    with LOG.open("a") as log:
        for name, command, additions in stages:
            started = time.time()
            row = {"name": name, "status": "RUNNING", "started": started}
            state["status"] = "RUNNING"
            state["current_stage"] = name
            state["stages"].append(row)
            atomic_status(state)
            log.write(f"\n[{name}] START {started}\n")
            log.flush()
            result = subprocess.run(
                command, cwd=ROOT,
                env={**os.environ, "HOME": str(ROOT), **additions},
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            row.update({
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "returncode": result.returncode,
                "elapsed_seconds": round(time.time() - started, 3),
            })
            atomic_status(state)
            if result.returncode != 0:
                state["status"] = "FAIL"
                state["failed_stage"] = name
                atomic_status(state)
                return result.returncode
    state.pop("current_stage", None)
    state["status"] = "SCALE_V1_AUTOMATIC_AND_GOLD_PACKAGE_PASS"
    state["completed_at"] = time.time()
    atomic_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
