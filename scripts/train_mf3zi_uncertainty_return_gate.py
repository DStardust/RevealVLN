#!/usr/bin/env python3
"""Fit MF3ZI's scene-disjoint one-shot uncertainty return/harm gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_rxr_uad_action_aligned_gate_mf3ze as base  # noqa: E402
from revealnav_mf3.uncertainty_gate import (  # noqa: E402
    FEATURE_NAMES,
    uncertainty_action_features,
)


base.SOURCE = ROOT / (
    "artifacts/phase1/mf3zi_uncertainty_direct_switch_returns_v1/"
    "MF3ZI_UNCERTAINTY_MANIFEST.json"
)
base.DESIGN = ROOT / (
    "artifacts/design/METHOD_FREEZE_3ZI_CAUSAL_UNCERTAINTY_ARBITRATION.md"
)
base.OUT = ROOT / "artifacts/training/mf3zi_uncertainty_return_gate_v1"
base.GATE = base.OUT / "MF3ZI_CROSSFIT_GATE.json"
base.MODEL = base.OUT / "MF3ZI_GATE_MODELS.npz"
base.SCHEMA_TAG = "mf3zi"
base.SOURCE_STATUS = "UNCERTAINTY_DIRECT_SWITCH_RETURN_DATASET_READY"
base.EXPECTED_ROWS = 126
base.EXPECTED_SCENES = 46
base.MIN_AUTHORIZED = 20


def feature_vector(row: dict, arrays: dict):
    vector = uncertainty_action_features(
        row["decision"], arrays["instruction"], arrays["checkpoint"],
        arrays["native"], arrays["alternative"],
    )
    return vector, list(FEATURE_NAMES)


base.feature_vector = feature_vector

# MF3ZI's online contract is semantic: an optional uncertainty switch must
# have non-negative robust predicted return.  The generic MF3ZE search also
# considers negative thresholds for its older proposal gate, so constrain this
# revision explicitly instead of relying on a post-hoc interpretation.
_candidate_rules = base.candidate_rules


def candidate_rules(expected, harm):
    for return_threshold, harm_threshold, mask in _candidate_rules(expected, harm):
        if return_threshold >= 0.0:
            yield return_threshold, harm_threshold, mask


base.candidate_rules = candidate_rules


if __name__ == "__main__":
    raise SystemExit(base.main())
