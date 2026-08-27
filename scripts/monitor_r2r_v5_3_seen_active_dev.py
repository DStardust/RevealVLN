#!/usr/bin/env python3
"""Read-only progress monitor for the 24-episode V5.3 paired dev gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_continuous_metric_v5_3_seen_active_dev"
PROTOCOL = OUT / "R2R_CONTINUOUS_METRIC_PROTOCOL_V5_3.json"
STATUS = OUT / "full/RUN_STATUS.json"
RUNS = OUT / "full/runs"


def process_counts() -> tuple[int, int]:
    runners = workers = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"run_r2r_continuous_metric_gate_v5_3.py" in command and b"seen-active-dev" in command:
            runners += 1
        if b"r2r_continuous_controller_worker_v5_3.py" in command and str(OUT).encode() in command:
            workers += 1
    return runners, workers


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    expected = int(protocol["runs"]["total"])
    rows = []
    for path in sorted(RUNS.glob("*/RUN_SUMMARY.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    passed = sum(row.get("status") == "PASS" for row in rows)
    failed = sum(row.get("status") == "FAIL" for row in rows)
    reveal = [row for row in rows if row.get("mode") == "revealnav"]
    excursions = sum(
        row.get("controller", {}).get("checkpointed_excursions", 0)
        for row in reveal
    )
    backtracks = sum(
        row.get("controller", {}).get("backtrack_decisions", 0)
        for row in reveal
    )
    returns = sum(
        row.get("controller", {}).get("successful_returns", 0)
        for row in reveal
    )
    runners, workers = process_counts()
    times = [
        float(row["wall_time_s"]) for row in rows
        if row.get("status") == "PASS" and row.get("wall_time_s") is not None
    ]
    average = sum(times) / len(times) if times else None
    eta = None if average is None else average * (expected - len(rows)) / max(1, workers)
    final = "RUNNING"
    if STATUS.is_file():
        try:
            final = json.loads(STATUS.read_text())["status"]
        except (OSError, json.JSONDecodeError, KeyError):
            final = "UNREADABLE"
    print("RevealNav V5.3 24 条活跃 episode 配对开发门")
    print(f"进度          {len(rows)}/{expected} ({100 * len(rows) / expected:.1f}%)")
    print(f"PASS/FAIL     {passed}/{failed}")
    print(f"RevealNav完成 {len(reveal)}/72")
    print(f"探索/回退/返回 {excursions}/{backtracks}/{returns}")
    print(f"后台          runner={runners}, workers={workers}")
    print(f"平均耗时      {duration(average)} / run")
    print(f"预计剩余      {duration(eta)}")
    print(f"最终状态      {final}")


if __name__ == "__main__":
    main()
