#!/usr/bin/env python3
"""Print one current V6 collection status snapshot."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="pilot_v6_0")
    args = parser.parse_args()
    root = ROOT / "artifacts/phase1/rxr_v6" / args.cohort
    progress = root / "RXR_V6_PAIR_PROGRESS.json"
    manifest = root / "RXR_V6_PAIRED_DATASET_MANIFEST.json"
    value = json.loads(progress.read_text()) if progress.is_file() else {
        "stage": "not_started", "completed": 0, "selected": 0,
        "remaining": 0, "active": {}, "failures": [],
    }
    if manifest.is_file():
        value["dataset"] = json.loads(manifest.read_text())["metadata"]
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
