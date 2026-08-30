#!/usr/bin/env python3
"""One-shot status for the complete V6 train-only campaign."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/phase1/rxr_v6"


def main() -> int:
    state = BASE / "RXR_V6_CAMPAIGN_STATE.json"
    value = json.loads(state.read_text()) if state.is_file() else {
        "status": "NOT_STARTED"
    }
    for cohort in ("pilot_v6_0", "full_v6_0"):
        progress = BASE / cohort / "RXR_V6_PAIR_PROGRESS.json"
        manifest = BASE / cohort / "RXR_V6_PAIRED_DATASET_MANIFEST.json"
        if progress.is_file():
            row = json.loads(progress.read_text())
            value[cohort] = {key: row.get(key) for key in (
                "stage", "selected", "completed", "remaining", "active",
                "failures",
            )}
        if manifest.is_file():
            value.setdefault(cohort, {})["dataset"] = json.loads(
                manifest.read_text()
            )["metadata"]
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
