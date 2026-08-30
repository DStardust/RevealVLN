#!/usr/bin/env python3
"""Focused tests for expanded BranchExcursion Q training contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rxr_branch_excursion_q_v5_1 as training  # noqa: E402
from revealnav_mf2r5 import BranchExcursionDataset  # noqa: E402


class ExpandedQTrainingTest(unittest.TestCase):
    def test_scene_partition_closure(self) -> None:
        train, development, counts = training.partitions()
        self.assertFalse(train & development)
        self.assertEqual((len(train), len(development)), (1498, 332))
        self.assertEqual((counts["train_scenes"], counts["development_scenes"]), (26, 9))

    def test_one_event_loads(self) -> None:
        event_id = next(iter(training.partitions()[0]))
        dataset = BranchExcursionDataset(training.MANIFEST, {event_id})
        row = dataset[0]
        self.assertTrue(row["candidate_mask"][-1].all())
        self.assertTrue(torch.isfinite(row["commit_cost"]).all())
        self.assertTrue(torch.isfinite(row["excursion_cost"]).all())

    def test_tie_aware_constant_is_order_independent(self) -> None:
        truth = torch.tensor([1.0, 3.0, 2.0, 4.0])
        prediction = torch.ones(4)
        a = training.tie_aware_cost(truth, prediction)
        b = training.tie_aware_cost(torch.flip(truth, (0,)), prediction)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
