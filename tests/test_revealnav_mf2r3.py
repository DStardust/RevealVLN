import unittest

import torch

from revealnav_mf2 import RevealOptionLossConfig
from revealnav_mf2r3 import (
    BalancedStructuredUADExpiryLoss,
    ExpiryQAdapterLoss,
    PairedQAdapterLoss,
    RelationalRevealExpiryHeads,
)


class ExpiryRevisionTest(unittest.TestCase):
    def test_independent_expiry_output_and_gradient(self) -> None:
        model = RelationalRevealExpiryHeads(8, 4, 2)
        history = torch.randn(2, 5, 8)
        candidates = torch.randn(2, 5, 3, 8)
        mask = torch.ones(2, 5, 3, dtype=torch.bool)
        budgets = torch.ones(2, 5, 2)
        instruction = torch.randn(2, 8)
        output = model(history, candidates, mask, budgets, instruction)
        self.assertEqual(output.expiry_hazard_logit.shape, (2, 5))
        self.assertTrue(torch.all(
            output.option_cost_without_checkpoint[mask]
            >= output.option_cost[mask]
        ))
        batch = {
            "candidate_mask": mask,
            "target_index": torch.zeros(2, 5, dtype=torch.long),
            "target_in_set": torch.ones(2, 5),
            "separation": torch.ones(2, 5),
            "evidence_complete": torch.ones(2, 5),
            "reveal_hazard": torch.zeros(2, 5),
            "expiry_hazard": torch.tensor([
                [0.0, 0.0, 1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            "option_cost": torch.ones(2, 5, 3),
            "option_cost_without_checkpoint": torch.full((2, 5, 3), 2.0),
            "current_feasibility": torch.ones(2, 5, 3, 2),
            "checkpoint_value": torch.zeros(2, 5),
        }
        objective = BalancedStructuredUADExpiryLoss(
            RevealOptionLossConfig(), (1.0, 1.0, 1.0)
        )
        losses = objective(output, batch)
        self.assertTrue(torch.isfinite(losses["expiry"]))
        losses["total"].backward()
        self.assertIsNotNone(model.event_heads.weight.grad)

    def test_adapter_loss_does_not_require_decision_labels(self) -> None:
        model = RelationalRevealExpiryHeads(8, 4, 2)
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        output = model(
            torch.randn(1, 3, 8), torch.randn(1, 3, 2, 8), mask,
            torch.ones(1, 3, 2), torch.randn(1, 8),
        )
        losses = ExpiryQAdapterLoss()(output, {
            "candidate_mask": mask,
            "option_cost_without_checkpoint": torch.ones(1, 3, 2),
            "expiry_hazard": torch.tensor([[0.0, 0.0, 1.0]]),
        })
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.expiry_head.weight.grad)
        self.assertIsNotNone(model.no_checkpoint_delta_head.weight.grad)

    def test_paired_q_adapter_trains_both_q_outputs(self) -> None:
        model = RelationalRevealExpiryHeads(8, 4, 2)
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        output = model(
            torch.randn(1, 3, 8), torch.randn(1, 3, 2, 8), mask,
            torch.ones(1, 3, 2), torch.randn(1, 8),
        )
        batch = {
            "candidate_mask": mask,
            "option_cost": torch.tensor([[[1., 2.], [2., 1.], [1., 3.]]]),
            "option_cost_without_checkpoint": torch.tensor(
                [[[2., 2.], [3., 1.], [2., 4.]]]
            ),
        }
        losses = PairedQAdapterLoss()(output, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.cost_head.weight.grad)
        self.assertIsNotNone(model.no_checkpoint_delta_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
