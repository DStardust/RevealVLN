#!/usr/bin/env python3
"""Authorize the frozen causal frontend for human-accepted queue50 episodes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cr5_causal_frontend_worker as worker  # noqa: E402


ACCEPTANCE = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/human_review_fast/"
    "CR5_QUEUE50_HUMAN_REVIEW_ACCEPTANCE.json"
)
value = json.loads(ACCEPTANCE.read_text())
if value.get("status") != "PASS":
    raise SystemExit("queue50 human review is not accepted")
worker.ALLOWED_EPISODES = {
    event_id.split("_ep", 1)[1].split("_", 1)[0]
    for event_id in value["accepted_event_ids"]
}


if __name__ == "__main__":
    raise SystemExit(worker.main())
