"""Mathematical and fail-closed invariants for MF3ZM-CAR v1."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.car import (
    CAR_POLICY_FEATURE_NAMES,
    catastrophic_rate_constraint,
    event_domain_weights,
    projected_dual_update,
    selected_utility_constraint,
    straight_through_gate,
)
from revealnav_mf3.car_fast import fit_car_ensemble_fast
from revealnav_mf3.car_selection import fit_car_ensemble


class CARModelTest(unittest.TestCase):
    def test_domain_event_weights_are_event_equal_and_domain_balanced(self):
        values = np.asarray(["RxR"] * 3 + ["R2R"] * 2)
        weights = event_domain_weights(values)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights[values == "RxR"].sum()), 0.5)
        self.assertAlmostEqual(float(weights[values == "R2R"].sum()), 0.5)
        self.assertTrue(np.all(weights[values == "RxR"] == weights[0]))

    def test_straight_through_forward_is_exact_hard_mask_and_has_gradient(self):
        logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        surrogate, hard, probability = straight_through_gate(logits)
        self.assertTrue(torch.equal(surrogate.detach(), hard))
        self.assertTrue(torch.equal(hard, torch.tensor([0.0, 0.0, 1.0])))
        surrogate.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad > 0))
        self.assertTrue(torch.isfinite(probability).all())

    def test_hard_risk_constraint_matches_event_rate(self):
        logits = torch.tensor([-1.0, 1.0, -2.0, 2.0], requires_grad=True)
        catastrophic = torch.tensor([0.0, 1.0, 0.0, 0.0])
        weights = torch.full((4,), 0.25)
        value, zero = catastrophic_rate_constraint(
            logits, catastrophic, weights, 0.25, hard_forward=True,
        )
        # The hard-selected rows are indices 1 and 3; one is catastrophic.
        self.assertFalse(zero)
        self.assertAlmostEqual(float(value.detach()), 0.25, places=6)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_selected_domain_is_explicitly_reported(self):
        logits = torch.tensor([-2.0, -1.0], requires_grad=True)
        catastrophic = torch.zeros(2)
        weights = torch.full((2,), 0.5)
        value, zero = catastrophic_rate_constraint(
            logits, catastrophic, weights, 0.0, hard_forward=True,
        )
        self.assertTrue(zero)
        self.assertEqual(float(value.detach()), 0.0)

    def test_scene_utility_constraint_matches_fixed_population_sum(self):
        logits = torch.tensor([-1.0, 1.0, 2.0], requires_grad=True)
        target = torch.tensor([0.4, -0.2, 0.3])
        weights = torch.full((3,), 1.0 / 3.0)
        subset = torch.tensor([True, True, False])
        value, zero = selected_utility_constraint(
            logits, target, weights, subset, hard_forward=True,
        )
        self.assertFalse(zero)
        # Only index 1 is selected in the subset, so -w*y = +0.2/3.
        self.assertAlmostEqual(float(value.detach()), 0.2 / 3.0, places=6)

    def test_dual_projection_is_nonnegative_and_capped(self):
        dual = torch.tensor(0.0)
        positive = projected_dual_update(dual, torch.tensor(2.0), 1.0, maximum=1.5)
        negative = projected_dual_update(positive, torch.tensor(-5.0), 1.0, maximum=1.5)
        self.assertEqual(float(positive), 1.5)
        self.assertEqual(float(negative), 0.0)

    def test_policy_feature_contract_is_fixed(self):
        self.assertEqual(len(CAR_POLICY_FEATURE_NAMES), 10)
        self.assertNotIn("scene_id", CAR_POLICY_FEATURE_NAMES)
        self.assertNotIn("dataset", CAR_POLICY_FEATURE_NAMES)

    def test_fast_executor_matches_reference_training(self):
        rng = np.random.default_rng(7)
        rows = 24
        inputs = {"policy_only": rng.normal(
            size=(rows, len(CAR_POLICY_FEATURE_NAMES))
        ).astype(np.float32)}
        target = rng.normal(scale=0.2, size=rows)
        scenes = np.asarray([f"scene-{index % 6}" for index in range(rows)])
        datasets = np.asarray([
            "RxR" if index % 2 else "R2R" for index in range(rows)
        ])
        episodes = np.asarray([f"episode-{index}" for index in range(rows)])
        kwargs = {
            "weight_decay": 0.001,
            "seeds": (11, 12, 13),
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "training_steps": 5,
            "dual_cap": 100.0,
            "representation": "policy_only",
            "risk_mode": "hard",
            "scene_constraint": True,
            "use_cuda": False,
        }
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            reference = fit_car_ensemble(
                inputs, target, scenes, datasets, episodes, **kwargs
            )
            accelerated = fit_car_ensemble_fast(
                inputs, target, scenes, datasets, episodes, **kwargs
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual(reference[1], accelerated[1])
        self.assertEqual(
            [value["hard_selected_by_domain"] for value in reference[2]],
            [value["hard_selected_by_domain"] for value in accelerated[2]],
        )
        maximum = max(
            float((value - accelerated[0][model_index].state_dict()[name])
                  .abs().max())
            for model_index, model in enumerate(reference[0])
            for name, value in model.state_dict().items()
        )
        self.assertLessEqual(maximum, 1e-7)


if __name__ == "__main__":
    unittest.main()
