from __future__ import annotations

import unittest

import torch

from revealnav_mf2.model import RevealOptionOutput
from revealnav_mf2r1 import factorized_uad_probabilities


class StructuredUADTest(unittest.TestCase):
    def test_factorized_probabilities_are_normalized_and_exact(self):
        logits = torch.zeros(1, 1)
        output = RevealOptionOutput(
            target_logits=torch.zeros(1, 1, 2),
            option_cost=torch.zeros(1, 1, 2),
            current_feasibility_logits=torch.zeros(1, 1, 2, 4),
            target_in_set_logit=logits,
            separation_logit=logits,
            evidence_logit=logits,
            reveal_hazard_logit=logits,
            checkpoint_value=logits,
        )
        probabilities = factorized_uad_probabilities(output)
        self.assertTrue(torch.allclose(
            probabilities, torch.tensor([[[0.5, 0.375, 0.125]]])
        ))
        self.assertTrue(torch.allclose(
            probabilities.sum(-1), torch.ones(1, 1)
        ))


if __name__ == "__main__":
    unittest.main()
