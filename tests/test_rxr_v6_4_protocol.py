#!/usr/bin/env python3

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_rxr_v6_4_hurdle_advantage as v64
from train_rxr_v6_relative_advantage import (
    earliest_authorized_policy,
    partition_indices,
)


class V64ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.arrays, cls.records = v64.load_inputs(
            v64.CANONICAL_MANIFEST.resolve()
        )
        cls.episodes = json.loads(v64.SELECTION.read_text())["episodes"]

    def test_hierarchical_weights_equalize_scenes_and_episodes(self):
        for fold in range(5):
            fit, _, _, _ = partition_indices(
                self.records, fold, "v6_3_1"
            )
            weights = v64.scene_episode_event_weights(
                fit, self.records, torch.device("cpu")
            ).numpy()
            by_scene = defaultdict(float)
            by_episode = defaultdict(float)
            episode_scene = {}
            for value, index in zip(weights, fit):
                row = self.records[int(index)]
                scene = str(row["scene_id"])
                episode = str(row["episode_id"])
                by_scene[scene] += float(value)
                by_episode[episode] += float(value)
                episode_scene[episode] = scene
            self.assertLess(max(by_scene.values()) - min(by_scene.values()), 1e-5)
            for scene in by_scene:
                values = [
                    value for episode, value in by_episode.items()
                    if episode_scene[episode] == scene
                ]
                self.assertLess(max(values) - min(values), 1e-5)

    def test_scene_isolation_and_complete_episode_accounting(self):
        accounted = 0
        zero_candidate = 0
        all_evaluation_scenes = set()
        for fold in range(5):
            fit, calibration, evaluation, _ = partition_indices(
                self.records, fold, "v6_3_1"
            )
            scene_sets = [
                {str(self.records[int(index)]["scene_id"]) for index in rows}
                for rows in (fit, calibration, evaluation)
            ]
            self.assertFalse(scene_sets[0] & scene_sets[1])
            self.assertFalse(scene_sets[0] & scene_sets[2])
            self.assertFalse(scene_sets[1] & scene_sets[2])
            self.assertFalse(all_evaluation_scenes & scene_sets[2])
            all_evaluation_scenes |= scene_sets[2]
            policy = earliest_authorized_policy(
                np.full(len(evaluation), -1.0),
                self.arrays["target"][evaluation], evaluation,
                self.records, self.episodes, fold,
            )
            accounted += policy["episodes"]
            zero_candidate += policy["zero_candidate_episodes"]
        self.assertEqual(accounted, 120)
        self.assertEqual(zero_candidate, 1)
        self.assertEqual(len(all_evaluation_scenes), 59)


if __name__ == "__main__":
    unittest.main()
