#!/usr/bin/env python3
"""Print one compact status snapshot for expanded V5 label generation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase1/rxr_train_expansion/branch_excursion_v5_1"
PROGRESS = OUT / "RXR_BRANCH_EXCURSION_LABEL_PROGRESS_V5_1.json"
GATE = OUT / "RXR_BRANCH_EXCURSION_LABEL_GATE_V5_1.json"


def main() -> int:
    value = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {
        "status": "NOT_STARTED", "total": 1830, "completed": 0, "failed": 0,
        "eta_s": None,
    }
    result = json.loads(GATE.read_text()) if GATE.exists() else None
    generated = len(list((OUT / "runs").glob("*.json")))
    new_total = 1406
    elapsed = float(value.get("elapsed_s") or 0.0)
    rate = generated / elapsed if elapsed > 0 and generated > 0 else 0.0
    active_workers = 0
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        active_workers += int(b"rxr_branch_excursion_label_worker_v5.py" in command)
    print(json.dumps({
        "state": value.get("status"),
        "population_completed": value.get("completed"),
        "population_total": value.get("total"),
        "accepted_v4_reused": 424,
        "newly_generated": generated,
        "new_generation_total": new_total,
        "new_generation_remaining": max(0, new_total - generated),
        "active_workers": active_workers,
        "failed": value.get("failed"),
        "eta_minutes": None if rate == 0
                       else round(max(0, new_total - generated) / rate / 60, 1),
        "final_gate": None if result is None else result.get("status"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
