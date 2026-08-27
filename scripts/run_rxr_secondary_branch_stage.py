#!/usr/bin/env python3
"""Issue one immutable branch-proposer response per rendered secondary event."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
INPUT = BASE / "multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json"
RUNNER = ROOT / "scripts/run_rxr_secondary_branch_factory.py"
AGGREGATOR = ROOT / "scripts/aggregate_rxr_secondary_first_response.py"
LOG_DIR = BASE / "branch_factory/worker_logs"
SHARD_COUNT = 16


def run_shard(shard: int) -> tuple[int, int]:
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
                "--execute",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return shard, result.returncode


def main() -> int:
    manifest = json.loads(INPUT.read_text())
    if manifest.get("status") != "READY_FOR_BRANCH_PROPOSER":
        raise SystemExit("secondary multiview input is not ready")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(run_shard, range(SHARD_COUNT)))
    # Exit 1 means at least one first response was semantically invalid.  That
    # is an expected fail-closed datum; the aggregator rejects it rather than
    # resampling the provider.
    unexpected = [item for item in results if item[1] not in (0, 1)]
    if unexpected:
        raise RuntimeError(f"branch worker process failures: {unexpected}")
    aggregate = subprocess.run(
        [sys.executable, str(AGGREGATOR)], cwd=ROOT, check=False
    )
    return aggregate.returncode


if __name__ == "__main__":
    raise SystemExit(main())
