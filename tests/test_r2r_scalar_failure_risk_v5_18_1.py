#!/usr/bin/env python3
"""Contract tests for the invariant scalar failure-risk model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from revealnav_scalar_failure_risk import (  # noqa: E402
    FEATURE_NAMES, ScalarETPFailureRiskHead, causal_scalar_features,
)


class ScalarFailureRiskTests(unittest.TestCase):
    def test_feature_and_logit_shapes(self) -> None:
        embeddings = [torch.randn(4, 768) for _ in range(5)]
        distance = torch.rand(4, 2)
        features = causal_scalar_features(*embeddings, distance)
        self.assertEqual(tuple(features.shape), (4, len(FEATURE_NAMES)))
        model = ScalarETPFailureRiskHead(
            torch.zeros(len(FEATURE_NAMES)), torch.ones(len(FEATURE_NAMES))
        )
        self.assertEqual(tuple(model(*embeddings, distance).shape), (4,))

    def test_identical_candidates_have_zero_contrast(self) -> None:
        embeddings = [torch.randn(2, 768) for _ in range(3)]
        candidate = torch.randn(2, 768)
        features = causal_scalar_features(
            *embeddings, candidate, candidate, torch.ones(2, 2)
        )
        self.assertTrue(torch.allclose(features[:, 10:13], torch.zeros(2, 3)))
        self.assertTrue(torch.allclose(features[:, 15], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
