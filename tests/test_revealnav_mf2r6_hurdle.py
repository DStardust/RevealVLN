#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6 import (
    FailureConditionedHurdleAdvantage,
    FailureConditionedHurdleLoss,
    V64_TRAINING_CONTRACT,
    validate_hurdle_checkpoint_payload,
)


class HurdleAdvantageTest(unittest.TestCase):
    def inputs(self):
        return [torch.randn(4, 8) for _ in range(6)]

    def test_failure_gate_ignores_alternative_identity(self):
        torch.manual_seed(3)
        model = FailureConditionedHurdleAdvantage(input_dim=8, projection_dim=4)
        values = self.inputs()
        scalars = torch.randn(4, 20)
        left = model(*values, scalars)
        values[-1] = torch.randn(4, 8) * 10.0
        changed_scalars = scalars.clone()
        changed_scalars[:, 4:6] += 100.0
        right = model(*values, changed_scalars)
        self.assertTrue(torch.equal(left.failure_logit, right.failure_logit))

    def test_return_distance_is_monotonic_penalty(self):
        torch.manual_seed(5)
        model = FailureConditionedHurdleAdvantage(input_dim=8, projection_dim=4)
        values = self.inputs()
        near = torch.randn(4, 20)
        near[:, 2] = 0.1
        far = near.clone()
        far[:, 2] = 0.8
        near_output = model(*values, near)
        far_output = model(*values, far)
        self.assertTrue(torch.all(far_output.failure_expert < near_output.failure_expert))
        self.assertTrue(torch.all(far_output.success_expert < near_output.success_expert))
        self.assertTrue(torch.all(
            far_output.expected_advantage < near_output.expected_advantage
        ))

    def test_locked_loss_is_finite_and_reaches_every_parameter(self):
        torch.manual_seed(7)
        model = FailureConditionedHurdleAdvantage(input_dim=8, projection_dim=4)
        output = model(*self.inputs(), torch.randn(4, 20))
        losses = FailureConditionedHurdleLoss(3.0, 3.0)(
            output, torch.tensor([0.2, -0.1, -0.2, -0.3]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.ones(4),
        )
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_checkpoint_validator_rejects_provenance_drift(self):
        model = FailureConditionedHurdleAdvantage()
        provenance = {"sealed": {"bytes": 1, "sha256": "a" * 64}}
        partition = {
            f"{role}_{suffix}": (
                "a" * 64 if suffix == "scene_ids_sha256" else 1
            )
            for role in ("fit", "calibration", "evaluation")
            for suffix in ("rows", "scene_count", "scene_ids_sha256")
        }
        losses = {
            key: 0.1
            for key in ("total", "mixture", "conditional", "failure", "sign")
        }
        effective_seeds = [
            seed + 200
            for seed in V64_TRAINING_CONTRACT["member_base_seeds"]
        ]
        payload = {
            "schema_version": "revealnav-rxr-v6.4-hurdle-ensemble/1",
            "method_revision": "v6_4_failure_conditioned_hurdle",
            "fold": 2,
            "member_base_seeds": V64_TRAINING_CONTRACT["member_base_seeds"],
            "member_effective_seeds": effective_seeds,
            "training_contract": V64_TRAINING_CONTRACT,
            "locked_provenance": provenance,
            "empirical_lower_offset": 0.1,
            "partition_evidence": partition,
            "training_evidence": [
                {
                    "effective_seed": seed,
                    "fit_rows": 1,
                    "failure_positive_weight": 1.0,
                    "sign_positive_weight": 1.0,
                    "final_fit_loss": losses,
                    "failure_probability_used_as_mixture_weight": True,
                    "failure_probability_used_as_independent_gate": False,
                    "sign_score_used_as_independent_gate": False,
                }
                for seed in effective_seeds
            ],
            "model_state_dicts": [model.state_dict() for _ in range(3)],
        }
        validate_hurdle_checkpoint_payload(payload, provenance, 2, partition)
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_hurdle_checkpoint_payload(
                payload, {"sealed": {"bytes": 2, "sha256": "b" * 64}},
                2, partition,
            )


if __name__ == "__main__":
    unittest.main()
