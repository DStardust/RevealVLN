#!/usr/bin/env python3
"""Print one concise status line for the MF3S RxR val_seen evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3s_uad_rxr_val_seen_v1"
PROGRESS = OUT / "MF3S_RXR_VAL_SEEN_PROGRESS.json"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    if not PROGRESS.is_file():
        print("MF3S val_seen: waiting for progress file")
        return
    value = json.loads(PROGRESS.read_text())
    total = int(value["total"])
    completed = int(value["completed"])
    elapsed = float(value["elapsed_s"])
    remaining = (
        elapsed / completed * (total - completed) if completed else None
    )
    print(
        f"{time.strftime('%F %T')}  status={value['status']}  "
        f"completed={completed}/{total}  failed={value['failed']}  "
        f"active={len(value['active'])}  queued={value['queued']}  "
        f"elapsed={duration(elapsed)}  eta={duration(remaining)}"
    )
    for row in value["active"]:
        print(
            f"  gpu={row['gpu']} mode={row['mode']} "
            f"episode={row['episode_id']}"
        )


if __name__ == "__main__":
    main()
