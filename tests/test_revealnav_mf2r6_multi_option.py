#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6 import (
    MultiOptionListwiseAdvantage,
    MultiOptionListwiseLoss,
    select_v65_authorized_option,
)


class MultiOptionAdvantageTest(unittest.TestCase):
    def inputs(self):
        shared = [torch.randn(3, 8) for _ in range(5)]
        alternatives = torch.randn(3, 3, 8)
        base = torch.randn(3, 1, 16)
        scalars = base.expand(-1, 3, -1).clone()
        scalars[..., 2] = torch.tensor([[0.1], [0.2], [0.3]])
        scalars[..., 4] = torch.tensor([
            [0.2, 0.3, 0.4], [0.1, 0.2, 0.0], [0.4, 0.3, 0.2]
        ])
        scalars[..., 5] = scalars[..., 4] - scalars[:, :1, 3]
        mask = torch.tensor([
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ])
        return shared, alternatives, scalars, mask

    def test_candidate_permutation_is_equivariant(self):
        torch.manual_seed(11)
        model = MultiOptionListwiseAdvantage(
            input_dim=8, projection_dim=4, hidden_dim=12
        )
        shared, alternatives, scalars, mask = self.inputs()
        left = model(*shared, alternatives, scalars, mask)
        order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(order)
        right = model(
            *shared, alternatives[:, order], scalars[:, order], mask[:, order]
        )
        for left_value, right_value in zip(left[:3], right[:3]):
            self.assertTrue(torch.allclose(
                left_value, right_value[:, inverse], atol=1e-6, rtol=1e-6
            ))
        self.assertTrue(torch.equal(
            left.listwise_logits[:, :1], right.listwise_logits[:, :1]
        ))
        self.assertTrue(torch.allclose(
            left.listwise_logits[:, 1:],
            right.listwise_logits[:, 1:][:, inverse],
            atol=1e-5, rtol=1e-6,
        ))

    def test_padding_in_first_slot_does_not_change_valid_options(self):
        torch.manual_seed(12)
        model = MultiOptionListwiseAdvantage(
            input_dim=8, projection_dim=4, hidden_dim=12
        )
        shared, alternatives, scalars, mask = self.inputs()
        alternatives = alternatives[1:2]
        scalars = scalars[1:2]
        mask = mask[1:2]
        reference = model(*[value[1:2] for value in shared], alternatives, scalars, mask)
        order = torch.tensor([2, 0, 1])
        moved_alternatives = alternatives[:, order].clone()
        moved_scalars = scalars[:, order].clone()
        moved_mask = mask[:, order]
        moved_alternatives[:, 0] = 99.0
        moved_scalars[:, 0] = -77.0
        moved_scalars[:, 0, 2] = 0.0
        moved = model(
            *[value[1:2] for value in shared],
            moved_alternatives, moved_scalars, moved_mask,
        )
        self.assertTrue(torch.allclose(
            reference.median[:, :2], moved.median[:, 1:],
            atol=1e-6, rtol=1e-6,
        ))
        self.assertEqual(moved.median[:, 0].item(), 0.0)

    def test_return_distance_is_only_a_monotonic_penalty(self):
        torch.manual_seed(13)
        model = MultiOptionListwiseAdvantage(
            input_dim=8, projection_dim=4, hidden_dim=12
        )
        shared, alternatives, near, mask = self.inputs()
        near[..., 2] = 0.1
        far = near.clone()
        far[..., 2] = 0.8
        near_output = model(*shared, alternatives, near, mask)
        far_output = model(*shared, alternatives, far, mask)
        for near_value, far_value in zip(near_output[:3], far_output[:3]):
            self.assertTrue(torch.all(
                far_value[mask] < near_value[mask]
            ))

    def test_loss_is_permutation_invariant_and_reaches_all_parameters(self):
        torch.manual_seed(17)
        model = MultiOptionListwiseAdvantage(
            input_dim=8, projection_dim=4, hidden_dim=12
        )
        shared, alternatives, scalars, mask = self.inputs()
        target = torch.tensor([
            [0.2, 0.2, -0.1], [-0.2, 0.3, 0.0], [0.1, -0.4, 0.0]
        ])
        weight = torch.tensor([1.0, 2.0, 1.5])
        objective = MultiOptionListwiseLoss()
        output = model(*shared, alternatives, scalars, mask)
        left = objective(output, target, mask, weight)
        order = torch.tensor([1, 2, 0])
        permuted = model(
            *shared, alternatives[:, order], scalars[:, order], mask[:, order]
        )
        right = objective(
            permuted, target[:, order], mask[:, order], weight
        )
        self.assertTrue(torch.allclose(left["total"], right["total"], atol=1e-6))
        left["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_exact_delegation_and_shared_scalar_fail_closed(self):
        lower = torch.tensor([[-0.1, -0.2], [0.2, 0.2], [0.1, 0.4]])
        mask = torch.tensor([[True, True], [True, True], [True, False]])
        self.assertEqual(
            select_v65_authorized_option(lower, mask).tolist(), [-1, 0, 0]
        )
        model = MultiOptionListwiseAdvantage(
            input_dim=8, projection_dim=4, hidden_dim=12
        )
        shared, alternatives, scalars, option_mask = self.inputs()
        scalars[0, 1, 8] += 0.1
        with self.assertRaisesRegex(ValueError, "shared scalar"):
            model(*shared, alternatives, scalars, option_mask)


if __name__ == "__main__":
    unittest.main()
