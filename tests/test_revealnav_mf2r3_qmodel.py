import unittest

import torch

from revealnav_mf2r3 import CausalPairedQAdapter


class PairedQModelTest(unittest.TestCase):
    def test_order_mask_and_causal_future_invariance(self):
        torch.manual_seed(3)
        model = CausalPairedQAdapter(8, 6, 16.0).eval()
        history = torch.randn(1, 5, 8)
        candidates = torch.randn(1, 5, 3, 8)
        mask = torch.tensor([[[1, 1, 0]] * 5], dtype=torch.bool)
        instruction = torch.randn(1, 8)
        first = model(history, candidates, mask, instruction)
        changed_history = history.clone()
        changed_history[:, 3:] = torch.randn_like(changed_history[:, 3:]) * 10
        changed_candidates = candidates.clone()
        changed_candidates[:, 3:] = torch.randn_like(changed_candidates[:, 3:]) * 10
        second = model(
            changed_history, changed_candidates, mask, instruction
        )
        self.assertTrue(torch.equal(
            first.q_with_checkpoint[:, :3], second.q_with_checkpoint[:, :3]
        ))
        valid = mask
        self.assertTrue(torch.all(
            first.q_without_checkpoint[valid] >= first.q_with_checkpoint[valid]
        ))
        self.assertTrue(torch.isinf(first.q_with_checkpoint[~valid]).all())


if __name__ == "__main__":
    unittest.main()
