#!/usr/bin/env python3
"""Contract tests for the fixed REE-Q policy holdout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_r2r_ree_q_v5_2_policy_holdout as holdout  # noqa: E402


class R2RReeQPolicyHoldoutTest(unittest.TestCase):
    def test_same_seed_pairs(self) -> None:
        pairs = holdout.checkpoint_pairs()
        self.assertEqual([row["seed"] for row in pairs], list(holdout.SEEDS))

    def test_fixed_formula_and_no_ensemble(self) -> None:
        value = holdout.protocol_value()["composition"]
        self.assertIn("5.0", value["fused_action_cost"])
        self.assertEqual(value["aggregation"], "report each seed independently; no ensemble")

    def test_no_new_threshold(self) -> None:
        value = holdout.protocol_value()
        self.assertEqual(value["composition"]["activation"], "candidate_score > 0; no calibrated threshold")
        self.assertFalse(value["val_seen_used_for_threshold_or_selection"])


if __name__ == "__main__":
    unittest.main()
