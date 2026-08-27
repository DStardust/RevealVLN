#!/usr/bin/env python3
"""Run eight isolated secondary causal-frontend shards on GPUs 0 through 7."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
RUNNER = ROOT / "scripts/run_rxr_secondary_frontend_shard.py"
LOG_DIR = BASE / "causal_frontend/worker_logs"
SHARD_COUNT = 8


def run_shard(shard: int) -> tuple[int, int]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(shard)
    log_path = LOG_DIR / f"shard_{shard:02d}.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--shard-index",
                str(shard),
                "--shard-count",
                str(SHARD_COUNT),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return shard, result.returncode


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=SHARD_COUNT) as executor:
        results = list(executor.map(run_shard, range(SHARD_COUNT)))
    failures = [item for item in results if item[1] != 0]
    if failures:
        raise RuntimeError(f"causal frontend shard process failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
