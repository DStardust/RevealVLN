"""Focused tests for the R2R zero-tuning transfer harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_uad_mf3zg_transfer as transfer  # noqa: E402


class R2RMF3ZGTransferTest(unittest.TestCase):
    def test_scene_selection_is_deterministic_and_balanced(self) -> None:
        rows = [
            {"episode_id": episode, "scene_id": f"mp3d/{scene}/{scene}.glb"}
            for scene in ("17DRP5sb8fy", "1LXtFkjw3qL")
            for episode in (1, 2, 3)
        ]
        first = transfer.select_one_per_scene(rows, "test-salt")
        second = transfer.select_one_per_scene(list(reversed(rows)), "test-salt")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len({row["scene_id"] for row in first}), 2)

    def test_scene_bootstrap_preserves_constant_delta(self) -> None:
        rows = [
            {"scene_id": scene, "utility": 0.25, "spl": 0.10}
            for scene in ("a", "b", "c")
        ]
        result = transfer.scene_cluster_bootstrap(
            rows, ("utility", "spl"), seed=7, replicates=100
        )
        self.assertEqual(result["utility"]["mean"], 0.25)
        self.assertEqual(
            result["utility"]["scene_bootstrap_95pct"], [0.25, 0.25]
        )
        self.assertAlmostEqual(result["spl"]["mean"], 0.10)


if __name__ == "__main__":
    unittest.main()
