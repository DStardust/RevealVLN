from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxr_scale_frontend_absorption_probe import (  # noqa: E402
    collapse_candidate_relations, reverse_candidates,
    swap_cross_event_candidates,
)


class FrontendAbsorptionInterventionTest(unittest.TestCase):
    def setUp(self):
        embeddings = torch.zeros(3, 3, 2, 1)
        embeddings[0, :, :, 0] = torch.tensor([[1, 3], [2, 4], [5, 7]])
        embeddings[1, :, :, 0] = 10
        embeddings[2, :, :, 0] = 20
        self.batch = {
            "candidate_embeddings": embeddings,
            "candidate_mask": torch.tensor([
                [[True, True], [True, True], [True, True]],
                [[True, True], [True, True], [False, False]],
                [[True, True], [False, False], [False, False]],
            ]),
            "step_mask": torch.tensor([
                [True, True, True], [True, True, False], [True, False, False],
            ]),
        }

    def test_reverse_moves_embeddings_and_masks_together(self):
        changed = reverse_candidates(self.batch)
        self.assertEqual(float(changed["candidate_embeddings"][0, 0, 0]), 3.0)
        self.assertTrue(torch.equal(
            changed["candidate_mask"], self.batch["candidate_mask"].flip(2)
        ))

    def test_collapse_preserves_mean_and_zeros_invalid_slots(self):
        changed = collapse_candidate_relations(self.batch)
        self.assertEqual(float(changed["candidate_embeddings"][0, 0, 0]), 2.0)
        self.assertEqual(float(changed["candidate_embeddings"][0, 0, 1]), 2.0)
        self.assertEqual(float(changed["candidate_embeddings"][1, 2].sum()), 0.0)

    def test_cross_event_swap_uses_different_donors_and_target_length(self):
        changed = swap_cross_event_candidates(self.batch)
        self.assertTrue(torch.equal(
            changed["candidate_embeddings"][0, :3, :, 0],
            torch.full((3, 2), 20.0),
        ))
        self.assertTrue(torch.equal(
            changed["candidate_embeddings"][1, :2, :, 0],
            torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        ))
        self.assertFalse(changed["candidate_mask"][1, 2].any())


if __name__ == "__main__":
    unittest.main()
