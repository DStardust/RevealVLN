import unittest

import torch

from revealnav_mf2r4 import BranchExcursionQHead, BranchExcursionQLoss


class BranchExcursionQTest(unittest.TestCase):
    def batch(self):
        torch.manual_seed(7)
        return {
            "history_embeddings": torch.randn(2, 5, 12),
            "candidate_embeddings": torch.randn(2, 5, 3, 12),
            "candidate_mask": torch.tensor([
                [[1, 1, 0]] * 5,
                [[1, 1, 1]] * 5,
            ], dtype=torch.bool),
            "instruction_embedding": torch.randn(2, 12),
            "decision_index": torch.tensor([3, 4]),
            "commit_cost": torch.tensor([[1.0, 6.0, torch.inf], [1.5, 6.5, 7.0]]),
            "excursion_cost": torch.tensor([[1.0, 3.0, torch.inf], [1.5, 3.5, 4.0]]),
        }

    def forward(self, model, batch):
        return model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["decision_index"],
        )

    def test_shapes_loss_and_backward(self):
        batch = self.batch()
        model = BranchExcursionQHead(12, 8, 16.0)
        output = self.forward(model, batch)
        self.assertEqual(output.commit_cost.shape, (2, 3))
        self.assertTrue(torch.isinf(output.commit_cost[0, 2]))
        losses = BranchExcursionQLoss()(output, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_candidate_permutation_equivariance(self):
        batch = self.batch()
        model = BranchExcursionQHead(12, 8, 16.0).eval()
        original = self.forward(model, batch)
        order = torch.tensor([2, 0, 1])
        permuted = dict(batch)
        permuted["candidate_embeddings"] = batch["candidate_embeddings"][:, :, order]
        permuted["candidate_mask"] = batch["candidate_mask"][:, :, order]
        result = self.forward(model, permuted)
        torch.testing.assert_close(result.commit_cost, original.commit_cost[:, order])
        torch.testing.assert_close(
            result.excursion_cost, original.excursion_cost[:, order]
        )


if __name__ == "__main__":
    unittest.main()
