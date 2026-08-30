#!/usr/bin/env python3
"""Background supervisor for the train-only MF3ZK collection and fitting.

It waits for the already-running R2R collection, assembles exact paired
returns, and starts the joint/ablation gate fit.  It never launches a public
split evaluation and writes one compact progress JSON for monitoring.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
COLLECTION_PROGRESS = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_COLLECTION_PROGRESS.json"
)
PIPELINE_PROGRESS = ROOT / "artifacts/training/mf3zk_joint_v1/MF3ZK_PIPELINE_PROGRESS.json"
LOG = ROOT / "artifacts/training/mf3zk_joint_v1/MF3ZK_PIPELINE.log"
BASELINE_SCRIPT = ROOT / "scripts/run_mf3zk_r2r_baseline_completion.py"
MANIFEST = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def write(value: dict) -> None:
    atomic_json(PIPELINE_PROGRESS, value)


def run_stage(name: str, command: list[str]) -> int:
    with LOG.open("a") as stream:
        stream.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {name}\n")
        stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            write({"status": "RUNNING", "stage": name, "updated_at": time.time()})
            time.sleep(10)
        stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END {name} rc={process.returncode}\n")
        return int(process.returncode)


def main() -> int:
    write({"status": "WAITING_FOR_COLLECTION", "stage": "collection", "updated_at": time.time()})
    while True:
        if COLLECTION_PROGRESS.is_file():
            try:
                collection = json.loads(COLLECTION_PROGRESS.read_text())
            except (OSError, ValueError):
                collection = {"status": "UNREADABLE"}
            write({
                "status": "WAITING_FOR_COLLECTION" if collection.get("status") == "RUNNING" else collection.get("status"),
                "stage": "collection", "collection": collection,
                "updated_at": time.time(),
            })
            if collection.get("status") in ("COMPLETE", "FAIL"):
                if collection.get("status") != "COMPLETE" or collection.get("failed", 0):
                    write({"status": "BLOCKED_COLLECTION_FAILURE", "stage": "collection", "collection": collection, "updated_at": time.time()})
                    return 2
                break
        time.sleep(30)

    rc = run_stage("baseline_completion", [
        str(PYTHON), str(BASELINE_SCRIPT), "run",
        "--gpus", "0,1", "--workers-per-gpu", "4", "--resume",
    ])
    if rc != 0:
        write({
            "status": "BLOCKED_BASELINE_COMPLETION_FAILURE",
            "stage": "baseline_completion", "returncode": rc,
            "updated_at": time.time(),
        })
        return rc
    manifest_ready = False
    if MANIFEST.is_file() and not MANIFEST.is_symlink():
        try:
            manifest_ready = json.loads(MANIFEST.read_text()).get("status") == "R2R_DIRECT_SWITCH_RETURN_DATASET_READY"
        except (OSError, ValueError):
            manifest_ready = False
    if manifest_ready:
        write({"status": "ASSEMBLY_ALREADY_COMPLETE", "stage": "assemble", "updated_at": time.time()})
    else:
        rc = run_stage("assemble", [
            str(PYTHON), "scripts/run_mf3zk_r2r_collection.py", "assemble",
        ])
        if rc != 0:
            write({"status": "BLOCKED_ASSEMBLY_INSUFFICIENT_OR_FAILED", "stage": "assemble", "returncode": rc, "updated_at": time.time()})
            return rc
    rc = run_stage("joint_and_controls_fit", [
        str(PYTHON), "scripts/train_mf3zk_joint_action_aligned_gate.py", "fit",
    ])
    write({
        "status": "TRAINING_COMPLETE" if rc == 0 else "TRAINING_FAILED",
        "stage": "joint_and_controls_fit", "returncode": rc,
        "updated_at": time.time(),
    })
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
