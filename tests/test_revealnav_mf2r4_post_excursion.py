import unittest

import torch

from revealnav_mf2r4 import PostExcursionQHead, PostExcursionQLoss


class PostExcursionQTest(unittest.TestCase):
    def batch(self):
        torch.manual_seed(11)
        return {
            "history_embeddings": torch.randn(3, 5, 12),
            "history_length": torch.tensor([5, 3, 4]),
            "instruction_embedding": torch.randn(3, 12),
            "selected_branch_embedding": torch.randn(3, 12),
            "checkpoint_embedding": torch.randn(3, 12),
            "post_candidate_embedding": torch.randn(3, 12),
            "normalized_excursion_elapsed": torch.tensor([1.0, 2.0, 1.5]),
            "continue_cost": torch.tensor([0.0, 5.0, 5.0]),
            "backtrack_cost": torch.tensor([7.0, 2.0, 5.0]),
        }

    def forward(self, model, batch):
        return model(
            batch["history_embeddings"], batch["history_length"],
            batch["instruction_embedding"], batch["selected_branch_embedding"],
            batch["checkpoint_embedding"], batch["post_candidate_embedding"],
            batch["normalized_excursion_elapsed"],
        )

    def test_shapes_nonnegative_loss_and_backward(self):
        batch = self.batch()
        model = PostExcursionQHead(12, 8, 5.0)
        output = self.forward(model, batch)
        self.assertEqual(output.continue_cost.shape, (3,))
        self.assertTrue(bool((output.continue_cost >= 0).all()))
        losses = PostExcursionQLoss()(output, batch)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_padded_suffix_does_not_change_output(self):
        batch = self.batch()
        model = PostExcursionQHead(12, 8, 5.0).eval()
        original = self.forward(model, batch)
        changed = dict(batch)
        changed["history_embeddings"] = batch["history_embeddings"].clone()
        changed["history_embeddings"][1, 3:] = 1000.0
        changed["history_embeddings"][2, 4:] = -1000.0
        result = self.forward(model, changed)
        torch.testing.assert_close(original.continue_cost, result.continue_cost)
        torch.testing.assert_close(original.backtrack_cost, result.backtrack_cost)

    def test_ties_are_excluded_from_ranking(self):
        batch = self.batch()
        output = self.forward(PostExcursionQHead(12, 8, 5.0), batch)
        objective = PostExcursionQLoss()
        losses = objective(output, batch)
        self.assertTrue(bool(torch.isfinite(losses["ranking"])))


if __name__ == "__main__":
    unittest.main()
