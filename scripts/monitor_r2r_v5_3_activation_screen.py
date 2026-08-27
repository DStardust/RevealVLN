#!/usr/bin/env python3
"""Read-only progress monitor for the V5.3 activation screen."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_3_activation_screen"
PROTOCOL = OUT / "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL.json"
STATUS = OUT / "full/RUN_STATUS.json"
RUNS = OUT / "full/runs"


def active_processes() -> list[tuple[int, str]]:
    marker = str(OUT).encode()
    script_markers = (
        b"run_r2r_v5_3_activation_screen.py run",
        b"run_r2r_v5_3_activation_screen.py resume",
        b"r2r_v5_3_activation_shadow_worker.py",
    )
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command or any(value in command for value in script_markers):
            rows.append((int(entry.name), command.decode(errors="replace").strip()))
    return sorted(rows)


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    if not PROTOCOL.is_file():
        raise SystemExit(f"protocol missing: {PROTOCOL}")
    expected = int(json.loads(PROTOCOL.read_text())["runs"])
    summaries = []
    for path in sorted(RUNS.glob("*/RUN_SUMMARY.json")):
        try:
            summaries.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    passed = sum(row.get("status") == "PASS" for row in summaries)
    failed = sum(row.get("status") == "FAIL" for row in summaries)
    active_episodes = sum(
        row.get("controller", {}).get("activation_count", 0) > 0
        for row in summaries if row.get("status") == "PASS"
    )
    total_activations = sum(
        row.get("controller", {}).get("activation_count", 0)
        for row in summaries if row.get("status") == "PASS"
    )
    workers = [
        row for row in active_processes()
        if "r2r_v5_3_activation_shadow_worker.py" in row[1]
    ]
    runners = [
        row for row in active_processes()
        if (
            "run_r2r_v5_3_activation_screen.py run" in row[1]
            or "run_r2r_v5_3_activation_screen.py resume" in row[1]
        )
    ]
    durations = [
        float(row["wall_time_s"]) for row in summaries
        if row.get("status") == "PASS" and row.get("wall_time_s") is not None
    ]
    average = sum(durations) / len(durations) if durations else None
    remaining = max(0, expected - len(summaries))
    parallelism = max(1, len(workers))
    eta = None if average is None else average * remaining / parallelism
    final_status = None
    if STATUS.is_file():
        try:
            final_status = json.loads(STATUS.read_text()).get("status")
        except (OSError, json.JSONDecodeError):
            final_status = "UNREADABLE"
    percent = 100.0 * len(summaries) / expected
    print("RevealNav V5.3 激活预筛")
    print(f"进度       {len(summaries)}/{expected} ({percent:.1f}%)")
    print(f"PASS/FAIL  {passed}/{failed}")
    print(f"活跃剧集   {active_episodes}（激活事件 {total_activations}）")
    print(f"后台       runner={len(runners)}, workers={len(workers)}")
    print(f"平均耗时   {duration(average)} / episode")
    print(f"预计剩余   {duration(eta)}")
    print(f"最终状态   {final_status or 'RUNNING'}")
    if failed:
        failed_ids = [
            str(row.get("episode_id")) for row in summaries
            if row.get("status") == "FAIL"
        ]
        print(f"失败 ID    {','.join(failed_ids)}")


if __name__ == "__main__":
    main()
