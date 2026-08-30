#!/usr/bin/env python3
"""Print the latest durable RxR V5.22 progress in one line."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / (
    "artifacts/evaluation/mf2_rxr_primary_v5_22_seen_dev/"
    "RXR_PRIMARY_SCREEN_PROGRESS_V5_22.json"
)
FULL = PATH.parent / "full" / "runs"
RESULT = PATH.parent / "RXR_PRIMARY_SCREEN_RESULT_V5_22.json"


def active_workers() -> int:
    count = 0
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = [
                value.decode(errors="replace")
                for value in cmdline.read_bytes().split(b"\0") if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(
            Path(value).name == "rxr_primary_controller_worker_v5_22.py"
            for value in argv
        ):
            count += 1
    return count


if RESULT.is_file():
    result = json.loads(RESULT.read_text())
    print(
        f"RxR V5.22: {result['status']} | 96/96 完成 | "
        f"checkpointed_excursions="
        f"{result['controller_activity']['checkpointed_excursions']}"
    )
elif FULL.is_dir():
    summaries = list(FULL.glob("*/RUN_SUMMARY.json"))
    passed = 0
    failed = 0
    for path in summaries:
        status = json.loads(path.read_text()).get("status")
        passed += status == "PASS"
        failed += status != "PASS"
    active = active_workers()
    print(
        f"RxR V5.22: RUNNING | {len(summaries)}/96 完成 | "
        f"PASS={passed} FAIL={failed} | 运行中={active} "
        f"排队≈{max(0, 96 - len(summaries) - active)}"
    )
elif not PATH.is_file():
    print("RxR V5.22: 尚未开始执行")
else:
    row = json.loads(PATH.read_text())
    print(
        f"RxR V5.22: {row['status']} | "
        f"{row['completed']}/{row['total']} 完成 | "
        f"PASS={row['passed']} FAIL={row['failed']} | "
        f"运行中={len(row['active'])} 排队={row['queued']} | "
        f"耗时={row['elapsed_s'] / 60:.1f} 分钟"
    )
