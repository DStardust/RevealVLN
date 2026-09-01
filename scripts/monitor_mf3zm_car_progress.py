#!/usr/bin/env python3
"""Print one compact, truthful MF3ZM-CAR work-unit progress bar."""

from __future__ import annotations

import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "artifacts/training/mf3zm_car_v1/MF3ZM_CAR_PROGRESS.json"
RESULT = ROOT / (
    "artifacts/training/mf3zm_car_v1/"
    "MF3ZM_CAR_TRAIN_DEVELOPMENT_RESULT.json"
)


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    if not PROGRESS.is_file():
        print("MF3ZM-CAR: progress file not created")
        return 1
    value = json.loads(PROGRESS.read_text())
    percent = float(value.get("progress_percent", 0.0))
    width = 40
    filled = min(width, max(0, int(percent * width / 100.0)))
    bar = "#" * filled + "-" * (width - filled)
    pid = int(value.get("pid", -1))
    alive = Path(f"/proc/{pid}").exists()
    elapsed = time.time() - float(value.get("started_unix", time.time()))
    print(
        f"MF3ZM-CAR [{bar}] {percent:6.2f}%  "
        f"{value.get('status')}  elapsed {duration(elapsed)}"
    )
    print(
        f"phase {value.get('phase_index')}/{value.get('phase_count')}: "
        f"{value.get('phase')}  |  ensemble fits "
        f"{value.get('phase_fit_completed')}/{value.get('phase_fit_maximum')}"
    )
    current = value.get("current_fit") or {}
    if current:
        print(
            f"current: {current.get('kind')} fit #{current.get('fit_number')}  "
            f"rows={current.get('rows')}  wd={current.get('weight_decay')}"
        )
    print(
        f"pid={pid} alive={str(alive).lower()}  result={RESULT.is_file()}  "
        f"message={value.get('message')}"
    )
    if RESULT.is_file():
        result = json.loads(RESULT.read_text())
        print(
            f"scientific result: {result.get('status')}  "
            f"checkpoint={result.get('checkpoint_created')}  "
            f"public_unseen={result.get('public_unseen_authorized')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
