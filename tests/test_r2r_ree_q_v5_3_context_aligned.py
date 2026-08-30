#!/usr/bin/env python3
"""Contract tests for train-supported causal context alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_r2r_ree_q_v5_3_context_aligned as context  # noqa: E402


class ContextAlignedFusionTest(unittest.TestCase):
    def test_cap_comes_from_training_distribution(self) -> None:
        value = context.context_contract()
        self.assertEqual(value["training_events"], 1830)
        self.assertEqual(value["context_cap"], 5)

    def test_no_threshold_or_weight_change(self) -> None:
        value = context.protocol_value()
        self.assertEqual(value["unchanged"]["wrong_commitment_weight"], 5.0)
        self.assertEqual(value["unchanged"]["activation"], "candidate_score > 0; no calibrated threshold")
        self.assertFalse(value["unchanged"]["ensemble"])

    def test_result_role_is_development(self) -> None:
        value = context.protocol_value()
        self.assertIn("development diagnostic", value["evaluation_role"])


if __name__ == "__main__":
    unittest.main()
