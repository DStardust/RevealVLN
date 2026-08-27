#!/usr/bin/env python3
"""Run and aggregate the 24-shard scale-v2 hindsight factory."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
RUNNER = ROOT / "scripts/run_rxr_scale_v2_hindsight_factory.py"
AGGREGATOR = ROOT / "scripts/aggregate_rxr_scale_v2_hindsight_factory.py"
STATUS = BASE / "RXR_SCALE_V2_HINDSIGHT_STATUS.json"
SHARDS = 24
GPUS = tuple(range(8))


def atomic_status(value: dict) -> None:
    part = STATUS.with_name(STATUS.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, STATUS)


def run_one(shard: int) -> dict:
    physical_gpu = GPUS[shard % len(GPUS)]
    log_dir = BASE / "hindsight_factory/worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard_{shard:02d}.log"
    started = time.time()
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--shard-index",
                str(shard),
                "--shard-count",
                str(SHARDS),
                "--gpu",
                "0",
                "--execute",
            ],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(physical_gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "shard": shard,
        "physical_gpu": physical_gpu,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> int:
    state = {
        "schema_version": "revealnav-rxr-scale-v2-hindsight-status/1",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started": time.time(),
        "expected_shards": SHARDS,
        "shards": [],
    }
    atomic_status(state)
    with ThreadPoolExecutor(max_workers=SHARDS) as executor:
        futures = [executor.submit(run_one, shard) for shard in range(SHARDS)]
        for future in as_completed(futures):
            row = future.result()
            state["shards"].append(row)
            state["completed"] = len(state["shards"])
            state["updated"] = time.time()
            atomic_status(state)
    failures = [row for row in state["shards"] if row["returncode"]]
    if failures:
        state.update({"status": "FAIL", "failures": failures})
        atomic_status(state)
        return 1
    result = subprocess.run(
        [sys.executable, str(AGGREGATOR)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    state["aggregate"] = {
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    state["status"] = "SCALE_V2_HINDSIGHT_PASS" if result.returncode == 0 else "FAIL"
    state["completed_at"] = time.time()
    atomic_status(state)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
