#!/usr/bin/env python3
"""Specialize the direct-return builder for MF3ZF's expanded train band."""

from __future__ import annotations

from pathlib import Path

import build_mf3zd_direct_switch_returns as builder


ROOT = Path(__file__).resolve().parents[1]
builder.SCHEMA_TAG = "mf3zf"
builder.WORKER_REVISION = "mf3zf"
builder.LOWER_QUANTILE = 0.970
builder.UPPER_QUANTILE = 0.995
builder.EXPECTED_EPISODES = 217
builder.EXPECTED_SCENES = 58
builder.MF3V_GATE = ROOT / (
    "artifacts/training/mf3zf_expanded_collection_v1/"
    "MF3ZF_COLLECTION_GATE.json"
)
builder.WORKER = ROOT / "scripts/rxr_uad_mf3zf_train_collection_worker.py"
builder.OUT = ROOT / "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1"
builder.SELECTION = builder.OUT / "MF3ZF_DIRECT_SWITCH_SELECTION.json"
builder.PROGRESS = builder.OUT / "MF3ZF_DIRECT_SWITCH_PROGRESS.json"
builder.MANIFEST = builder.OUT / "MF3ZF_DIRECT_SWITCH_MANIFEST.json"


if __name__ == "__main__":
    raise SystemExit(builder.main())
