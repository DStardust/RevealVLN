#!/usr/bin/env python3
"""Run one frozen causal-frontend shard for scale-v2."""

from pathlib import Path

import run_rxr_expansion_frontend_shard as frontend

ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/automatic"
frontend.GEOMETRY = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_GEOMETRY.json"
frontend.CONTROLLER = BASE / "multibranch/RXR_SCALE_MULTIBRANCH_CONTROLLER.json"
frontend.INPUTS = BASE / "multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
frontend.OUT_DIR = BASE / "causal_frontend"
frontend.SHARD_DIR = frontend.OUT_DIR / "frontend_shards"
frontend.RUN_DIR = frontend.OUT_DIR / "runs"

selected_events = frontend.selected_events


def unique_episode_events(shard_index: int, shard_count: int):
    events, sources = selected_events(shard_index, shard_count)
    unique = {}
    for event in events:
        unique.setdefault(event["episode_id"], event)
    return sorted(unique.values(), key=lambda row: row["expansion_order"]), sources


frontend.selected_events = unique_episode_events


if __name__ == "__main__":
    raise SystemExit(frontend.main())
