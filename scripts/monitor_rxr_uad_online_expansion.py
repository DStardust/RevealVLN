#!/usr/bin/env python3
"""Print one concise status line for the persistent MF3G collector."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / (
    "artifacts/phase1/mf3g_uad_online_expanded/dataset_v1/"
    "MF3B_ONLINE_DATA_PROGRESS.json"
)


def duration(seconds) -> str:
    if seconds is None:
        return "pending"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


if not PROGRESS.is_file():
    raise SystemExit("MF3G collector has not written progress yet")
value = json.loads(PROGRESS.read_text())
print(
    f"{value['status']} total={value['completed']}/{value['total']} "
    f"new={value.get('new_completed', 0)}/{value.get('new_total', 0)} "
    f"active={len(value['active'])} failed={value['failed']} "
    f"elapsed={duration(value.get('elapsed_s'))} eta={duration(value.get('eta_s'))}"
)
for slot, row in sorted(value["active"].items()):
    print(f"gpu:worker={slot} episode={row['episode_id']} scene={row['scene_id']}")
