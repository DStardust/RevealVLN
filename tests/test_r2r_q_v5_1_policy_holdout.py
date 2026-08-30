#!/usr/bin/env python3
"""Contract tests for the R2R policy-induced expanded-Q holdout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_r2r_q_v5_1_policy_holdout as holdout  # noqa: E402


class R2RQPolicyHoldoutTest(unittest.TestCase):
    def test_source_population(self) -> None:
        source = holdout.source_manifest()
        self.assertEqual(len(source["records"]), 576)
        self.assertEqual(sum(row["better_by_margin"] for row in source["records"]), 30)

    def test_three_source_balanced_checkpoints(self) -> None:
        rows = holdout.checkpoints()
        self.assertEqual([row["seed"] for row in rows], list(holdout.SEEDS))

    def test_protocol_has_no_tuned_threshold(self) -> None:
        value = holdout.protocol_value()
        self.assertEqual(value["model"]["activation"], "candidate_score > 0; no calibrated threshold")
        self.assertFalse(value["val_seen_used_for_threshold_or_selection"])


if __name__ == "__main__":
    unittest.main()
