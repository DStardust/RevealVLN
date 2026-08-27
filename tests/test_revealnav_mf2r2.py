from __future__ import annotations

import unittest

import torch

from revealnav_mf2r2 import RelationalRevealOptionHeads


class RelationalModelTest(unittest.TestCase):
    def test_variable_candidates_and_padded_step(self):
        model = RelationalRevealOptionHeads(8, 4, 2)
        mask = torch.tensor([[[True, True, False], [False, False, False]]])
        output = model(
            torch.randn(1, 2, 8),
            torch.randn(1, 2, 3, 8),
            mask,
            torch.ones(1, 2, 2),
            torch.randn(1, 8),
        )
        self.assertEqual(output.target_logits.shape, (1, 2, 3))
        self.assertEqual(output.current_feasibility_logits.shape, (1, 2, 3, 2))
        self.assertTrue(torch.isfinite(output.target_in_set_logit).all())
        self.assertTrue(torch.isfinite(output.separation_logit).all())
        self.assertTrue(torch.isneginf(output.target_logits[0, 1]).all())

    def test_stable_candidate_count_is_batch_partition_invariant(self):
        torch.manual_seed(7)
        model = RelationalRevealOptionHeads(
            8, 4, 2, candidate_count_encoding="saturating"
        ).eval()
        history = torch.randn(1, 2, 8)
        candidates = torch.randn(1, 2, 2, 8)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        budgets = torch.ones(1, 2, 2)
        instruction = torch.randn(1, 8)
        alone = model(history, candidates, mask, budgets, instruction)

        padded_candidates = torch.zeros(2, 2, 3, 8)
        padded_candidates[0, :, :2] = candidates[0]
        padded_candidates[1] = torch.randn(2, 3, 8)
        padded_mask = torch.ones(2, 2, 3, dtype=torch.bool)
        padded_mask[0, :, 2] = False
        together = model(
            torch.cat((history, torch.randn(1, 2, 8))),
            padded_candidates,
            padded_mask,
            torch.ones(2, 2, 2),
            torch.cat((instruction, torch.randn(1, 8))),
        )
        self.assertTrue(torch.allclose(
            alone.target_in_set_logit[0],
            together.target_in_set_logit[0],
            atol=1e-6, rtol=0.0,
        ))
        self.assertTrue(torch.allclose(
            alone.separation_logit[0],
            together.separation_logit[0],
            atol=1e-6, rtol=0.0,
        ))


if __name__ == "__main__":
    unittest.main()
