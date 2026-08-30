#!/usr/bin/env python3
"""Contracts for V5.19 alternative-first option search."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r2r_alternative_first_option_search_worker_v5_19 import (  # noqa: E402
    AlternativeFirstOptionSearchController,
)


class AlternativeFirstOptionSearchTests(unittest.TestCase):
    def test_transaction_budget_is_inherited(self) -> None:
        self.assertTrue(
            AlternativeFirstOptionSearchController._has_switch_budget(11, 15)
        )
        self.assertFalse(
            AlternativeFirstOptionSearchController._has_switch_budget(12, 15)
        )


if __name__ == "__main__":
    unittest.main()
