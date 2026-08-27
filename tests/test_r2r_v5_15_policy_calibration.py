#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_r2r_v5_15_policy_threshold import select_threshold  # noqa: E402
from run_r2r_v5_15_policy_calibration_pipeline import SEEDS, rows  # noqa: E402


class PolicyCalibrationTest(unittest.TestCase):
    def test_selection_crosses_every_episode_with_locked_seeds(self) -> None:
        selection = {
            "episodes": [
                {"episode_id": "1", "trajectory_id": "a", "scene_id": "s"},
                {"episode_id": "2", "trajectory_id": "b", "scene_id": "s"},
            ]
        }
        result = rows(selection)
        self.assertEqual(len(result), 6)
        self.assertEqual({row["controller_seed"] for row in result}, set(SEEDS))

    def test_threshold_requires_five_and_selects_positive_tail(self) -> None:
        scores = np.arange(100, dtype=np.float32)
        records = [{
            "event_id": f"e{index}",
            "alternative_branch_id": "a",
            "realized_trial_net_m": 2.0 if index >= 95 else -2.0,
            "better_by_margin": index >= 95,
        } for index in range(100)]
        threshold, policy = select_threshold(records, scores)
        self.assertEqual(threshold, 94.0)
        self.assertEqual(policy["activated"], 5)
        self.assertEqual(policy["positive_precision"], 1.0)
        self.assertGreater(policy["mean_net_per_event_m"], 0.0)

    def test_threshold_fails_when_five_activations_are_impossible(self) -> None:
        scores = np.arange(4, dtype=np.float32)
        records = [{
            "event_id": f"e{index}", "alternative_branch_id": "a",
            "realized_trial_net_m": 1.0, "better_by_margin": True,
        } for index in range(4)]
        with self.assertRaisesRegex(RuntimeError, "insufficient finite"):
            select_threshold(records, scores)


if __name__ == "__main__":
    unittest.main()
