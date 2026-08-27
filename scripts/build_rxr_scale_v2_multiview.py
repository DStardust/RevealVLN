#!/usr/bin/env python3
"""Render or aggregate scale-v2 automatic multi-view inputs."""

import json
from pathlib import Path

import build_rxr_scale_v1_multiview as base


ROOT = Path("/mnt/daiyang/vla")
SCALE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
SELECTION = SCALE / "RXR_SCALE_V2_SELECTION.json"
HINDSIGHT = SCALE / (
    "hindsight_factory/RXR_SCALE_V2_HINDSIGHT_EVENT_CANDIDATES.json"
)
QUEUE = SCALE / "RXR_SCALE_V2_ROUTE_CENSUS.json"

base.SCALE = SCALE
base.SELECTION = SELECTION
base.HINDSIGHT = HINDSIGHT
base.QUEUE = QUEUE


def documents(lane: str) -> tuple[dict, dict]:
    selection = json.loads(SELECTION.read_text())
    hindsight = json.loads(HINDSIGHT.read_text())
    if not (
        lane == "automatic"
        and selection.get("status") == "SCALE_V2_SELECTION_FROZEN"
        and hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
    ):
        raise RuntimeError("scale-v2 multiview precondition failed")
    return selection, {
        row["hindsight_candidate_id"]: row for row in hindsight["candidates"]
    }


base.documents = documents


if __name__ == "__main__":
    raise SystemExit(base.main())
