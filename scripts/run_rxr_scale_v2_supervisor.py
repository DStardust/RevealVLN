#!/usr/bin/env python3
"""Durably execute scale-v2 after scale-v1 releases its API and GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla")
V1 = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
STATUS = BASE / "RXR_SCALE_V2_SUPERVISOR_STATUS.json"
LOG = BASE / "RXR_SCALE_V2_SUPERVISOR.log"
CAUSAL_WORKERS = os.environ.get("RXR_SCALE_CAUSAL_WORKERS", "48")
STAGES = (
    ("hindsight_route_census", [sys.executable, "scripts/run_rxr_scale_v2_hindsight_stage.py"], {}),
    ("freeze_candidate_selection", [sys.executable, "scripts/build_rxr_scale_v2_selection.py"], {}),
    ("multiview", [sys.executable, "scripts/run_rxr_scale_v2_multiview.py"], {}),
    ("branch_first_responses", [sys.executable, "scripts/run_rxr_scale_v2_branch_stage.py"], {}),
    ("geometry", [sys.executable, "scripts/run_rxr_scale_v2_geometry.py"], {}),
    (
        "controller",
        [sys.executable, "scripts/run_rxr_scale_v2_controller.py"],
        {"CUDA_VISIBLE_DEVICES": "0", "CR5_CONTROLLER_GPU": "0"},
    ),
    ("causal_frontend", [sys.executable, "scripts/run_rxr_scale_v2_frontend_stage.py"], {}),
    ("causal_analysis", [sys.executable, "scripts/analyze_rxr_scale_v2_causal.py"], {}),
    (
        "causal_media",
        [sys.executable, "scripts/build_rxr_scale_v2_causal_media.py"],
        {"CUDA_VISIBLE_DEVICES": "0", "CR5_CAUSAL_MEDIA_GPU": "0"},
    ),
    (
        "causal_language",
        [sys.executable, "scripts/run_rxr_scale_v2_causal_language.py", "--execute", "--workers", CAUSAL_WORKERS],
        {},
    ),
    ("training_index", [sys.executable, "scripts/build_rxr_scale_v2_training_index.py"], {}),
    (
        "resource_labels",
        [sys.executable, "scripts/run_rxr_scale_v2_tx.py", "--gpus", "0,1,2,3,4,5,6,7", "--gpu-slots", "0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,3,4,5,6,7"],
        {},
    ),
    (
        "frozen_features",
        [sys.executable, "scripts/run_rxr_scale_v2_features.py", "--gpus", "0,1,2,3,4,5,6,7", "--gpu-slots", "0,0,0,1,1,1,2,3,4,5,6,7"],
        {},
    ),
    ("finalize_capacity", [sys.executable, "scripts/finalize_rxr_scale_v2_expansion.py"], {}),
)


def preflight() -> dict:
    try:
        workers = int(CAUSAL_WORKERS)
    except ValueError as error:
        raise RuntimeError("invalid causal-language worker count") from error
    if workers < 1:
        raise RuntimeError("causal-language worker count must be positive")
    scripts = [
        (ROOT / command[1]).resolve()
        for _, command, _ in STAGES
    ]
    missing = [str(path) for path in scripts if not path.is_file()]
    if missing:
        raise RuntimeError("missing scale-v2 stage scripts: " + ", ".join(missing))
    return {
        "causal_language_workers": workers,
        "stage_count": len(STAGES),
        "all_stage_scripts_present": True,
    }


def atomic_status(value: dict) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def wait_for_v1(state: dict) -> None:
    path = V1 / "RXR_SCALE_V1_SUPERVISOR_STATUS.json"
    gate = V1 / "automatic/multibranch/RXR_SCALE_FEATURE_GATE.json"
    while True:
        if path.is_file():
            value = json.loads(path.read_text())
            complete = value.get("status") == "SCALE_V1_AUTOMATIC_AND_GOLD_PACKAGE_PASS"
            expected_capacity_stop = (
                value.get("status") == "FAIL"
                and value.get("failed_stage") == "finalize_scale_capacity"
            )
            if complete or expected_capacity_stop:
                if not gate.is_file() or json.loads(gate.read_text()).get("status") != "FEATURE_GATE_PASS_AUTOMATIC_SCALE_READY":
                    raise RuntimeError("scale-v1 did not close its automatic feature gate")
                state["scale_v1_release"] = (
                    "COMPLETE" if complete else "EXPECTED_CAPACITY_SHORTFALL"
                )
                atomic_status(state)
                return
            if value.get("status") == "FAIL":
                raise RuntimeError(
                    "scale-v1 failed before capacity adjudication: "
                    + str(value.get("failed_stage"))
                )
        state["status"] = "WAITING_FOR_SCALE_V1_RESOURCE_RELEASE"
        state["updated"] = time.time()
        atomic_status(state)
        time.sleep(20)


def main() -> int:
    resume_stage = os.environ.get("RXR_SCALE_V2_RESUME_STAGE")
    if resume_stage:
        names = [row[0] for row in STAGES]
        if resume_stage not in names or not STATUS.is_file():
            raise RuntimeError("invalid scale-v2 resume stage")
        state = json.loads(STATUS.read_text())
        failed_resume = (
            state.get("status") == "FAIL"
            and state.get("failed_stage") == resume_stage
        )
        planned_resume = (
            os.environ.get("RXR_SCALE_V2_ALLOW_PLANNED_RESUME") == "1"
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
                "status": "INTERRUPTED_DURING_EXCEPTION_DRAIN",
                "interrupted_at": time.time(),
            })
        stages = STAGES[names.index(resume_stage):]
        state.update({
            "status": "RESUMING",
            "pid": os.getpid(),
            "resumed_at": time.time(),
            "resume_stage": resume_stage,
        })
        state.pop("failed_stage", None)
        state.pop("error", None)
    else:
        state = {
            "schema_version": "revealnav-rxr-scale-v2-supervisor-status/1",
            "status": "STARTING",
            "pid": os.getpid(),
            "started": time.time(),
            "stages": [],
        }
        stages = STAGES
    try:
        state["preflight"] = preflight()
    except Exception as error:
        state.update({"status": "FAIL", "error": repr(error)})
        atomic_status(state)
        return 1
    atomic_status(state)
    try:
        wait_for_v1(state)
    except Exception as error:
        state.update({"status": "FAIL", "error": repr(error)})
        atomic_status(state)
        return 1
    with LOG.open("a") as log:
        for name, command, additions in stages:
            started = time.time()
            row = {"name": name, "status": "RUNNING", "started": started}
            state.update({"status": "RUNNING", "current_stage": name})
            state["stages"].append(row)
            atomic_status(state)
            log.write(f"\n[{name}] START {started}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "HOME": str(ROOT), **additions},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            row.update(
                {
                    "status": "PASS" if result.returncode == 0 else "FAIL",
                    "returncode": result.returncode,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            atomic_status(state)
            if result.returncode:
                state.update({"status": "FAIL", "failed_stage": name})
                atomic_status(state)
                return result.returncode
    state.pop("current_stage", None)
    state["status"] = "SCALE_V2_EVENT_EXPANSION_PASS_GOLD_REVIEWS_REQUIRED"
    state["completed_at"] = time.time()
    atomic_status(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
