#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rxr_v6_1_counterfactual_worker import BroadPersistentCandidateController


class RxrV61CandidateTest(unittest.TestCase):
    def test_highest_non_native_is_selected(self):
        value = {"probabilities": torch.tensor([0.3, 0.6, 0.1])}
        selected = BroadPersistentCandidateController.ranked_alternative(
            value, ("native", "b", "c"), "native"
        )
        self.assertEqual(selected, "b")

    def test_tie_is_stable(self):
        value = {"probabilities": torch.tensor([0.2, 0.4, 0.4])}
        selected = BroadPersistentCandidateController.ranked_alternative(
            value, ("native", "z", "a"), "native"
        )
        self.assertEqual(selected, "a")


if __name__ == "__main__":
    unittest.main()
