#!/usr/bin/env python3
"""Read-only live monitor for the full val_seen V5.3 activation extension."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2"
PROTOCOL = OUT / "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL_V2.json"
STATUS = OUT / "full/RUN_STATUS.json"
RUNS = OUT / "full/runs"
V1_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen/"
    "R2R_V5_3_ACTIVATION_SCREEN_RESULT.json"
)


def process_counts() -> tuple[int, int]:
    runners = workers = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"run_r2r_v5_3_activation_screen_v2.py" in command:
            runners += 1
        if b"r2r_v5_3_activation_shadow_worker.py" in command and str(OUT).encode() in command:
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
    expected = int(json.loads(PROTOCOL.read_text())["runs"])
    v1 = json.loads(V1_RESULT.read_text())
    rows = []
    for path in sorted(RUNS.glob("*/RUN_SUMMARY.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    passed = sum(row.get("status") == "PASS" for row in rows)
    failed = sum(row.get("status") == "FAIL" for row in rows)
    v2_active = sum(
        row.get("controller", {}).get("activation_count", 0) > 0
        for row in rows if row.get("status") == "PASS"
    )
    events = sum(
        row.get("controller", {}).get("activation_count", 0)
        for row in rows if row.get("status") == "PASS"
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
    cumulative_active = int(v1["active_episodes"]) + v2_active
    cumulative_screened = int(v1["screened_episodes"]) + len(rows)
    cumulative_rate = cumulative_active / cumulative_screened if cumulative_screened else 0.0
    print("RevealNav V5.3 全量 val_seen 激活预筛 V2")
    print(f"V2 进度      {len(rows)}/{expected} ({100 * len(rows) / expected:.1f}%)")
    print(f"V2 PASS/FAIL {passed}/{failed}")
    print(f"累计进度     {cumulative_screened}/778")
    print(f"累计活跃     {cumulative_active} ({100 * cumulative_rate:.2f}%)")
    print(f"V2 激活事件  {events}")
    print(f"后台          runner={runners}, workers={workers}")
    print(f"平均耗时      {duration(average)} / episode")
    print(f"预计剩余      {duration(eta)}")
    print(f"最终状态      {final}")


if __name__ == "__main__":
    main()
