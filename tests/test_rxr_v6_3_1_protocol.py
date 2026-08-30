#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r6.protocol import outer_scene_partition, scene_fold
from train_rxr_v6_relative_advantage import earliest_authorized_policy


class V631ProtocolTest(unittest.TestCase):
    def test_scene_partition_is_total_and_disjoint(self):
        scenes = [f"scene-{index}" for index in range(40)]
        roles = outer_scene_partition(scenes, 2)
        self.assertEqual(set(roles), set(scenes))
        self.assertEqual(
            {scene for scene, role in roles.items() if role == "evaluation"},
            {scene for scene in scenes if scene_fold(scene) == 2},
        )
        self.assertGreaterEqual(sum(role == "calibration" for role in roles.values()), 1)
        self.assertGreaterEqual(sum(role == "fit" for role in roles.values()), 1)

    def test_only_earliest_authorized_event_counts(self):
        records = [
            {"row_index": 0, "episode_id": "a", "scene_id": "s",
             "post_navigation_step": 3, "event_id": "a3"},
            {"row_index": 1, "episode_id": "a", "scene_id": "s",
             "post_navigation_step": 5, "event_id": "a5"},
            {"row_index": 2, "episode_id": "b", "scene_id": "s",
             "post_navigation_step": 4, "event_id": "b4"},
        ]
        value = earliest_authorized_policy(
            np.asarray([0.1, 0.3, -0.2]),
            np.asarray([0.2, -0.5, 0.7]),
            np.asarray([0, 1, 2]), records,
            [
                {"episode_id": "a", "scene_id": "s", "scene_fold": 0},
                {"episode_id": "b", "scene_id": "s", "scene_fold": 0},
                {"episode_id": "c", "scene_id": "t", "scene_fold": 0},
            ],
            0,
        )
        self.assertEqual(value["selected_event_ids"], ["a3"])
        self.assertEqual(value["selected_episodes"], 1)
        self.assertEqual(value["episodes"], 3)
        self.assertEqual(value["zero_candidate_episodes"], 1)
        self.assertAlmostEqual(value["selected_realized_benefit_sum"], 0.2)
        self.assertAlmostEqual(value["scene_macro_policy_benefit"], 0.05)


if __name__ == "__main__":
    unittest.main()
