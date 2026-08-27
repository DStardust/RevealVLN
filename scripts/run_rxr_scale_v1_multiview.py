#!/usr/bin/env python3
"""Run scale-v1 multiview shards on eight GPUs and aggregate both lanes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla")
SCRIPT = ROOT / "scripts/build_rxr_scale_v1_multiview.py"
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
STATUS = BASE / "RXR_SCALE_V1_MULTIVIEW_STATUS.json"
SHARDS = 16
GPUS = tuple(range(8))


def atomic_status(value: dict) -> None:
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def run_one(lane: str, shard: int) -> dict:
    physical_gpu = GPUS[shard % len(GPUS)]
    log_dir = BASE / lane / "multiview/worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard_{shard:02d}.log"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(physical_gpu)}
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--lane", lane,
                "--shard-index", str(shard),
                "--shard-count", str(SHARDS),
                "--gpu", "0",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "lane": lane,
        "shard": shard,
        "physical_gpu": physical_gpu,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "revealnav-rxr-scale-multiview-status/1",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "shards": [],
        "expected": {"automatic": SHARDS, "new_gold": SHARDS},
    }
    atomic_status(state)
    jobs = [(lane, shard) for lane in ("automatic", "new_gold")
            for shard in range(SHARDS)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            state["shards"].append(result)
            state["completed"] = len(state["shards"])
            state["failed"] = sum(row["returncode"] != 0 for row in state["shards"])
            state["updated"] = time.time()
            atomic_status(state)
    failures = [row for row in state["shards"] if row["returncode"] != 0]
    if failures:
        state["status"] = "FAIL"
        state["failures"] = failures
        atomic_status(state)
        return 1
    aggregates = []
    for lane in ("automatic", "new_gold"):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--lane", lane,
             "--shard-count", str(SHARDS), "--aggregate"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        aggregates.append({
            "lane": lane,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        })
    state["aggregates"] = aggregates
    state["status"] = (
        "SCALE_V1_MULTIVIEW_PASS"
        if all(row["returncode"] == 0 for row in aggregates)
        else "FAIL"
    )
    state["completed_at"] = time.time()
    atomic_status(state)
    return 0 if state["status"] == "SCALE_V1_MULTIVIEW_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
