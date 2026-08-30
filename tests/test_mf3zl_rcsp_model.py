from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.rcsp import (
    EngineeredRCSPControl,
    POLICY_FEATURE_NAMES,
    RelativeSemanticSwitchPolicy,
    catastrophic_constraint,
    projected_dual_update,
    utility_weighted_preference_loss,
)


class RCSPModelTest(unittest.TestCase):
    def test_engineered_control_is_fixed_28d(self):
        model = EngineeredRCSPControl()
        value = model(torch.zeros(2, 28))
        self.assertEqual(tuple(value.shape), (2,))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 27))

    def test_forward_shape_finite_and_fixed_rank(self):
        torch.manual_seed(3)
        model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
        values = [torch.randn(4, 768) for _ in range(4)]
        logits = model(torch.randn(4, len(POLICY_FEATURE_NAMES)), *values)
        self.assertEqual(logits.shape, (4,))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES), rank=8)

    def test_large_positive_utility_has_larger_gradient_weight(self):
        logits = torch.zeros(2, requires_grad=True)
        loss = utility_weighted_preference_loss(
            logits, torch.tensor([0.01, 0.50]), torch.ones(2)
        )
        loss.backward()
        self.assertGreater(abs(float(logits.grad[1])), abs(float(logits.grad[0])))

    def test_negative_and_catastrophic_examples_push_toward_abstention(self):
        logits = torch.zeros(1, requires_grad=True)
        preference = utility_weighted_preference_loss(
            logits, torch.tensor([-0.2]), torch.ones(1)
        )
        risk = catastrophic_constraint(
            logits, torch.ones(1), torch.ones(1), ungated_rate=0.1
        )
        (preference + risk).backward()
        self.assertGreater(float(logits.grad), 0.0)

    def test_constraint_sign_and_domain_independence_inputs(self):
        safe = catastrophic_constraint(
            torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]),
            torch.ones(2), 0.1,
        )
        violated = catastrophic_constraint(
            torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]),
            torch.ones(2), 0.1,
        )
        self.assertLess(float(safe), 0.0)
        self.assertGreater(float(violated), 0.0)

    def test_dual_is_projected_and_only_violation_increases_it(self):
        zero = torch.tensor(0.0)
        self.assertEqual(
            float(projected_dual_update(zero, torch.tensor(-0.2), 0.1)), 0.0
        )
        self.assertGreater(
            float(projected_dual_update(zero, torch.tensor(0.2), 0.1)), 0.0
        )

    def test_zero_boundary_is_the_deployment_contract(self):
        logits = torch.tensor([-1e-6, 0.0, 1e-6])
        self.assertEqual((logits > 0).tolist(), [False, False, True])

    def test_zero_norm_semantic_difference_fails_closed(self):
        model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
        policy = torch.zeros(1, len(POLICY_FEATURE_NAMES))
        embedding = torch.ones(1, 768)
        with self.assertRaises(ValueError):
            model(policy, embedding, embedding, embedding, embedding)


if __name__ == "__main__":
    unittest.main()
