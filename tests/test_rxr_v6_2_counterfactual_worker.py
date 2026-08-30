#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rxr_v6_2_counterfactual_worker import LocalTopologyCandidateController


class RxrV62CandidateTest(unittest.TestCase):
    def test_native_and_three_oldest_are_retained(self):
        current = {key: object() for key in ("n", "a", "b", "c", "d")}
        ages = {"n": 1, "a": 2, "b": 5, "c": 5, "d": 3}
        self.assertEqual(
            LocalTopologyCandidateController.proposal_controls(
                current, "n", ages
            ),
            ("n", "b", "c", "d"),
        )

    def test_stop_or_no_alternative_fails_closed(self):
        self.assertEqual(
            LocalTopologyCandidateController.proposal_controls(
                {"a": object()}, None, {"a": 1}
            ),
            (),
        )
        self.assertEqual(
            LocalTopologyCandidateController.proposal_controls(
                {"a": object()}, "a", {"a": 1}
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
