from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.rcsp_v1_1 import (  # noqa: E402
    POLICY_FEATURE_NAMES,
    RelativeSemanticSwitchPolicy,
)


class RCSPV11ZeroDeltaTest(unittest.TestCase):
    def test_zero_relative_delta_is_retained_as_neutral_semantic_evidence(self):
        model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
        policy = torch.zeros(2, len(POLICY_FEATURE_NAMES))
        instruction = torch.ones(2, 768)
        history = torch.ones(2, 768) * 2
        native = torch.ones(2, 768)
        runner = native.clone()
        runner[1, 0] += 1.0
        result = model(policy, instruction, history, native, runner)
        self.assertEqual(tuple(result.shape), (2,))
        self.assertTrue(torch.isfinite(result).all())

    def test_absolute_zero_embedding_still_fails_closed(self):
        model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
        policy = torch.zeros(1, len(POLICY_FEATURE_NAMES))
        nonzero = torch.ones(1, 768)
        with self.assertRaises(ValueError):
            model(policy, torch.zeros_like(nonzero), nonzero, nonzero, nonzero)

    def test_zero_delta_does_not_disable_policy_side_channel(self):
        model = RelativeSemanticSwitchPolicy(len(POLICY_FEATURE_NAMES))
        with torch.no_grad():
            model.policy_head.weight.fill_(1.0)
        policy = torch.ones(2, len(POLICY_FEATURE_NAMES))
        embedding = torch.ones(2, 768)
        result = model(policy, embedding, embedding, embedding, embedding)
        self.assertNotEqual(float(result[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
