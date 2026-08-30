#!/usr/bin/env python3
"""Method-contract tests for V5.16 native-first delayed commitment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r2r_native_first_deferred_switch_worker_v5_16 import (  # noqa: E402
    NativeFirstDeferredSwitchController,
)


class NativeFirstDecisionTests(unittest.TestCase):
    def test_three_backtrack_votes_authorize_return(self) -> None:
        votes, allowed = (
            NativeFirstDeferredSwitchController._ensemble_backtrack_votes([
                (2.0, 1.0), (1.5, 1.4), (3.0, 0.5),
            ])
        )
        self.assertEqual(votes, (True, True, True))
        self.assertTrue(allowed)

    def test_one_disagreement_vetoes_return(self) -> None:
        votes, allowed = (
            NativeFirstDeferredSwitchController._ensemble_backtrack_votes([
                (2.0, 1.0), (1.0, 1.0), (3.0, 0.5),
            ])
        )
        self.assertEqual(votes, (True, False, True))
        self.assertFalse(allowed)

    def test_wrong_ensemble_width_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            NativeFirstDeferredSwitchController._ensemble_backtrack_votes([
                (2.0, 1.0), (3.0, 0.5),
            ])


if __name__ == "__main__":
    unittest.main()
