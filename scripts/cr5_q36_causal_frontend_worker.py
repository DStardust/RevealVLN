#!/usr/bin/env python3
"""Authorize the frozen causal frontend for corrected queue50 event q36."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_causal_frontend_worker as worker  # noqa: E402


CONTROLLER = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/"
    "CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json"
)
EXPECTED_SHA256 = (
    "e835f838e3111c497643d598db2f434d36bb76a01790b92a3cdc8ff0c2f878df"
)
if (not CONTROLLER.is_file() or CONTROLLER.is_symlink()
        or worker.sha256_file(CONTROLLER) != EXPECTED_SHA256):
    raise SystemExit("corrected controller evidence drift")
value = json.loads(CONTROLLER.read_text())
if (value.get("status_counts")
        != {"CONTROLLER_PASS_CAUSAL_GATE_REQUIRED": 1}
        or [row.get("event_id") for row in value.get("events", [])]
        != ["q36_ep1049_hv05"]):
    raise SystemExit("q36 controller gate did not pass exactly once")
worker.ALLOWED_EPISODES = {"1049"}


if __name__ == "__main__":
    raise SystemExit(worker.main())
