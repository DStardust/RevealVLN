#!/usr/bin/env python3
"""Run eight isolated scale-v2 causal-frontend shards."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
RUNNER = ROOT / "scripts/run_rxr_scale_v2_frontend_shard.py"
LOG_DIR = BASE / "causal_frontend/worker_logs"
SHARDS = 8


def run_one(shard: int) -> tuple[int, int]:
    path = LOG_DIR / f"shard_{shard:02d}.log"
    with path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--shard-index",
                str(shard),
                "--shard-count",
                str(SHARDS),
            ],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return shard, result.returncode


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=SHARDS) as executor:
        results = list(executor.map(run_one, range(SHARDS)))
    failures = [row for row in results if row[1]]
    if failures:
        raise RuntimeError(f"scale-v2 frontend shard failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
