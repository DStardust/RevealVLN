#!/usr/bin/env python3
"""Run the remaining secondary multiview shards on physical GPUs 0 and 1."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPT = ROOT / "scripts/build_rxr_secondary_multiview_factory.py"
OUT_DIR = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1/"
    "multiview_factory"
)
LOG_DIR = OUT_DIR / "worker_logs"
SHARD_COUNT = 16


def run_worker(physical_gpu: int, shards: tuple[int, ...]) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    for shard in shards:
        log_path = LOG_DIR / f"shard_{shard:02d}.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--shard-index",
                    str(shard),
                    "--shard-count",
                    str(SHARD_COUNT),
                    "--gpu",
                    "0",
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"render shard {shard} failed with exit {result.returncode}"
            )


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_worker, 0, (2, 4, 6, 8, 10, 12, 14)),
            executor.submit(run_worker, 1, (1, 3, 5, 7, 9, 11, 13, 15)),
        ]
        for future in futures:
            future.result()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--aggregate",
            "--shard-count",
            str(SHARD_COUNT),
        ],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
