#!/usr/bin/env python3
"""Small contracts for the V5.18 risk-gated option-search worker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r2r_risk_gated_option_search_worker_v5_18 import (  # noqa: E402
    RiskGatedOptionSearchController,
)


class RiskGatedOptionSearchTests(unittest.TestCase):
    def test_inherits_frozen_remaining_set_controller(self) -> None:
        self.assertTrue(issubclass(
            RiskGatedOptionSearchController,
            __import__(
                "r2r_remaining_set_rerank_worker_v5_17"
            ).RemainingSetRerankController,
        ))

    def test_transaction_budget_is_unchanged(self) -> None:
        self.assertTrue(RiskGatedOptionSearchController._has_switch_budget(11, 15))
        self.assertFalse(RiskGatedOptionSearchController._has_switch_budget(12, 15))


if __name__ == "__main__":
    unittest.main()
