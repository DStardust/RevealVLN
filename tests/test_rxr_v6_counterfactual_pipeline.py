#!/usr/bin/env python3

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rxr_v6_counterfactual_pipeline as pipeline
import rxr_v6_counterfactual_worker as worker


class RxrV6CounterfactualPipelineTest(unittest.TestCase):
    def test_only_evidenced_unexecutable_macro_is_a_rejection(self):
        base = {
            "status": "REJECTED_UNEXECUTABLE_MACRO",
            "mode": "macro",
            "split": "train",
            "metrics": {"ndtw": 0.5},
            "target_reached": True,
            "target_return_scheduled": True,
            "target_alternative_committed": False,
            "unseen_or_test_read": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(base))
            self.assertTrue(pipeline.rejected_macro_summary(path))
            path.write_text(json.dumps({**base, "target_reached": False}))
            self.assertFalse(pipeline.rejected_macro_summary(path))

    def test_scene_balanced_selection_is_deterministic(self):
        first = pipeline.select_episodes("unit_v6", 60)
        second = pipeline.select_episodes("unit_v6", 60)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 60)
        self.assertEqual(len({row["episode_id"] for row in first}), 60)
        self.assertEqual(len({row["scene_id"] for row in first}), 59)
        self.assertTrue(all(row["scene_fold"] in range(5) for row in first))

    def test_task_utility_exact_definition(self):
        metrics = {"ndtw": 0.4, "sdtw": 0.2, "spl": 0.6}
        self.assertAlmostEqual(pipeline.task_utility(metrics), 0.4)

    def test_repeated_gpu_indices_create_multiple_worker_slots(self):
        self.assertEqual(pipeline.parse_gpus("0,0,1,1,2"), (0, 0, 1, 1, 2))

    def test_causal_array_hash_uses_content(self):
        arrays = {
            "a": np.asarray([1, 2], dtype=np.float16),
            "b": np.asarray([3], dtype=np.float32),
        }
        original = worker.stable_array_hash(arrays)
        copied = worker.stable_array_hash({key: value.copy() for key, value in arrays.items()})
        changed = worker.stable_array_hash({**arrays, "b": np.asarray([4], dtype=np.float32)})
        self.assertEqual(original, copied)
        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
