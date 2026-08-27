import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from revealnav_net_advantage import PairwiseNetAdvantageHead
from run_r2r_train_net_advantage_pipeline import canonical_routes, pilot_routes
from train_r2r_sparse_net_advantage import event_policy


class PairwiseNetAdvantageHeadTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = PairwiseNetAdvantageHead()
        embeddings = [torch.randn(3, 768) for _ in range(5)]
        logits, gain = model(*embeddings, torch.rand(3, 2))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(gain.shape), (3,))
        self.assertTrue(bool((gain >= 0).all()))
        (logits.mean() + gain.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


class SparsePolicyTest(unittest.TestCase):
    def test_selects_at_most_one_alternative_per_event(self) -> None:
        records = [
            {"event_id": "a", "realized_trial_net_m": -2.0, "better_by_margin": False},
            {"event_id": "a", "realized_trial_net_m": 1.0, "better_by_margin": True},
            {"event_id": "b", "realized_trial_net_m": 0.5, "better_by_margin": True},
        ]
        result = event_policy(records, torch.tensor([0.1, 0.8, 0.2]).numpy(), 0.15)
        self.assertEqual(result["events"], 2)
        self.assertEqual(result["activated"], 2)
        self.assertEqual(result["selected_indices"], [1, 2])
        self.assertEqual(result["positive_precision"], 1.0)


class TrainSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.routes = canonical_routes()

    def test_one_episode_per_train_trajectory(self) -> None:
        self.assertEqual(len(self.routes), 3603)
        self.assertEqual(len({row["trajectory_id"] for row in self.routes}), 3603)
        self.assertEqual(len({row["scene_id"] for row in self.routes}), 61)

    def test_pilot_is_scene_diverse_and_deterministic(self) -> None:
        first = pilot_routes(self.routes)
        second = pilot_routes(self.routes)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 96)
        self.assertEqual(len({row["episode_id"] for row in first}), 96)
        self.assertGreaterEqual(len({row["scene_id"] for row in first}), 50)
        self.assertTrue(all(row["reference_points"] >= 5 for row in first))


if __name__ == "__main__":
    unittest.main()
