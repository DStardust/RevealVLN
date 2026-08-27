#!/usr/bin/env python3
"""Run and aggregate branch-proposer calls for both scale-v1 lanes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
RUNNER = ROOT / "scripts/run_rxr_scale_v1_branch_factory.py"
AGGREGATOR = ROOT / "scripts/aggregate_rxr_scale_v1_branch_first_response.py"
STATUS = BASE / "RXR_SCALE_V1_BRANCH_STATUS.json"
SHARDS = 32


def atomic_status(value: dict) -> None:
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def run_one(lane: str, shard: int) -> dict:
    log_dir = BASE / lane / "branch_factory/worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard_{shard:02d}.log"
    environment = {**os.environ, "RXR_SCALE_LANE": lane}
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--shard-index", str(shard),
             "--shard-count", str(SHARDS), "--execute"],
            cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    return {
        "lane": lane,
        "shard": shard,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> int:
    state = {
        "schema_version": "revealnav-rxr-scale-branch-status/1",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "expected_shards": 2 * SHARDS,
        "shards": [],
    }
    atomic_status(state)
    jobs = [(lane, shard) for lane in ("automatic", "new_gold")
            for shard in range(SHARDS)]
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            state["shards"].append(result)
            state["completed"] = len(state["shards"])
            state["updated"] = time.time()
            atomic_status(state)
    unexpected = [row for row in state["shards"]
                  if row["returncode"] not in (0, 1)]
    if unexpected:
        state.update({"status": "FAIL", "failures": unexpected})
        atomic_status(state)
        return 1
    aggregates = []
    for lane in ("automatic", "new_gold"):
        environment = {**os.environ, "RXR_SCALE_LANE": lane}
        result = subprocess.run(
            [sys.executable, str(AGGREGATOR)], cwd=ROOT, env=environment,
            capture_output=True, text=True, check=False,
        )
        aggregates.append({
            "lane": lane,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        })
    state["aggregates"] = aggregates
    state["status"] = (
        "SCALE_V1_BRANCH_PASS_WITH_FAIL_CLOSED_REJECTIONS"
        if all(row["returncode"] == 0 for row in aggregates)
        else "FAIL"
    )
    state["completed_at"] = time.time()
    atomic_status(state)
    return 0 if state["status"].startswith("SCALE_V1_BRANCH_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
