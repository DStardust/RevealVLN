#!/usr/bin/env python3
"""Specialize the action-aligned safety trainer for expanded MF3ZF coverage."""

from __future__ import annotations

from pathlib import Path

import train_rxr_uad_action_aligned_gate_mf3ze as trainer


ROOT = Path(__file__).resolve().parents[1]
trainer.SOURCE = ROOT / (
    "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/"
    "MF3ZF_DIRECT_SWITCH_MANIFEST.json"
)
trainer.DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZF_COVERAGE_SAFETY.md"
trainer.OUT = ROOT / "artifacts/training/mf3zf_action_aligned_return_gate_v1"
trainer.GATE = trainer.OUT / "MF3ZF_CROSSFIT_GATE.json"
trainer.MODEL = trainer.OUT / "MF3ZF_GATE_MODELS.npz"
trainer.SCHEMA_TAG = "mf3zf"
trainer.EXPECTED_ROWS = 217
trainer.EXPECTED_SCENES = 58
trainer.MIN_AUTHORIZED = 24


if __name__ == "__main__":
    raise SystemExit(trainer.main())
