import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from revealnav_net_advantage import (
    ONLINE_SCORE_DEFINITION,
    OnlineNetAdvantageScorer,
    PairwiseNetAdvantageHead,
)
from evaluate_r2r_v5_13_paired import paired_comparison, validate_training_result
from run_r2r_train_net_advantage_pipeline import canonical_routes, pilot_routes
from train_r2r_sparse_net_advantage import event_policy, predict


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

    def test_online_checkpoint_and_causal_score(self) -> None:
        model = PairwiseNetAdvantageHead()
        payload = {
            "schema_version": "revealnav-pairwise-net-advantage-checkpoint/1",
            "model_state_dict": model.state_dict(),
            "calibrated_score_threshold": -100.0,
            "score_definition": ONLINE_SCORE_DEFINITION,
            "immediate_cost_scale_m": 10.0,
            "input_dim": 768,
            "projection_dim": 96,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)
            scorer = OnlineNetAdvantageScorer.from_checkpoint(path)
        vector = torch.zeros(768)
        rows = scorer.score_candidates(
            vector, vector, vector, vector,
            {"b": vector, "a": vector}, 1.0, {"a": 2.0, "b": 3.0},
        )
        self.assertEqual([row["branch_id"] for row in rows], ["a", "b"])
        self.assertTrue(all(row["online_wrong_trial_cost_m"] in (4.0, 6.0) for row in rows))
        best = max(rows, key=lambda row: (row["net_advantage_score_m"], row["branch_id"]))
        self.assertTrue(scorer.approve(best["branch_id"], rows))

    def test_legacy_offline_threshold_fails_closed(self) -> None:
        model = PairwiseNetAdvantageHead()
        payload = {
            "schema_version": "revealnav-pairwise-net-advantage-checkpoint/1",
            "model_state_dict": model.state_dict(),
            "calibrated_score_threshold": 0.0,
            "score_definition": "offline_round_trip_cost",
            "immediate_cost_scale_m": 10.0,
            "input_dim": 768,
            "projection_dim": 96,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "offline-only"):
                OnlineNetAdvantageScorer.from_checkpoint(path)


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

    def test_score_uses_online_distance_not_offline_label_cost(self) -> None:
        class FixedModel:
            def __call__(self, *unused):
                return torch.zeros(1), torch.ones(1)

        arrays = {
            "immediate_costs": torch.tensor([[1.0, 2.0]]).numpy(),
            "round_trip_cost": torch.tensor([999.0]).numpy(),
        }
        batch = {
            key: torch.zeros(1, 768) for key in (
                "instruction", "current_history", "temporal_history", "native",
                "alternative",
            )
        }
        batch["immediate_costs"] = torch.zeros(1, 2)
        _, _, score = predict(FixedModel(), batch, arrays, torch.tensor([0]).numpy())
        self.assertAlmostEqual(float(score[0]), 3.0)


class PairedEvaluationTest(unittest.TestCase):
    def test_episode_mean_across_three_seeds(self) -> None:
        treatment = {}
        baseline = {}
        for episode in ("a", "b"):
            for seed in (20260826, 20260827, 20260828):
                baseline[(episode, seed)] = {
                    "metrics": {"spl": 0.4, "distance_to_goal": 4.0}
                }
                treatment[(episode, seed)] = {
                    "metrics": {"spl": 0.5, "distance_to_goal": 3.0}
                }
        result = paired_comparison(
            treatment, baseline, ["spl", "distance_to_goal"], 1000
        )
        self.assertEqual(result["paired_episodes"], 2)
        self.assertAlmostEqual(
            result["benefit_treatment_minus_baseline"]["spl"]["mean"], 0.1
        )
        self.assertAlmostEqual(
            result["benefit_treatment_minus_baseline"]["distance_to_goal"]["mean"],
            1.0,
        )

    def test_training_result_gate_fails_closed(self) -> None:
        value = {
            "status": "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_FAIL",
            "unseen_or_test_read": False,
            "task_metric_payload_read": False,
            "results": [
                {"seed": seed} for seed in (20260826, 20260827, 20260828)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(__import__("json").dumps(value))
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                validate_training_result(path)


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
