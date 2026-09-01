from __future__ import annotations

import unittest

import numpy as np
import torch

from revealnav_mf3.mf3zu_evidence_memory_reranker import (
    EvidenceMemoryResidualReranker,
    common_initialized_rerankers,
    fit_feature_normalizer,
    mean_pool_evidence,
    parameter_sha256,
    shuffled_memory_donor_indices,
)


class MF3ZUEvidenceMemoryRerankerTest(unittest.TestCase):
    def test_candidate_specific_mean_pool_and_budget(self):
        values = torch.zeros(2, 3, 8, 4)
        mask = torch.zeros(2, 3, 8, dtype=torch.bool)
        values[0, 0, 0] = 2.0
        values[0, 0, 1] = 4.0
        mask[0, 0, :2] = True
        values[0, 1, 0] = -3.0
        mask[0, 1, 0] = True
        pooled = mean_pool_evidence(values, mask)
        self.assertEqual(tuple(pooled.shape), (2, 3, 4))
        torch.testing.assert_close(pooled[0, 0], torch.full((4,), 3.0))
        torch.testing.assert_close(pooled[0, 1], torch.full((4,), -3.0))
        torch.testing.assert_close(pooled[1], torch.zeros(3, 4))
        with self.assertRaisesRegex(ValueError, "K_MEM=8"):
            mean_pool_evidence(torch.zeros(1, 2, 7, 4), torch.zeros(1, 2, 7, dtype=torch.bool))

    def test_reranker_detaches_frozen_etp_inputs_and_masks_padding(self):
        model = EvidenceMemoryResidualReranker(78)
        candidate = torch.randn(2, 3, 768, requires_grad=True)
        base = torch.randn(2, 3, requires_grad=True)
        memory = torch.randn(2, 3, 78, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        scores = model(candidate, base, mask, memory)
        self.assertTrue(torch.isneginf(scores[0, 2]))
        scores[mask].sum().backward()
        self.assertIsNone(candidate.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(memory.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, "candidate-specific"):
            model(candidate.detach(), base.detach(), mask, torch.randn(2, 78))

    def test_common_initialization_is_byte_identical(self):
        first, second, digest = common_initialized_rerankers(78)
        self.assertEqual(parameter_sha256(first), digest)
        self.assertEqual(parameter_sha256(second), digest)

    def test_shuffled_control_deranges_train_and_uses_train_for_held(self):
        ids = [f"event-{index}" for index in range(12)]
        counts = [2, 2, 3, 3, 4, 4, 1, 1, 2, 3, 4, 1]
        candidate_counts = [2, 2, 3, 3, 4, 4, 2, 2, 2, 3, 4, 2]
        train = np.arange(8)
        held = np.arange(8, 12)
        donors, diagnostic = shuffled_memory_donor_indices(
            ids, counts, train, held, candidate_counts=candidate_counts
        )
        self.assertTrue(np.all(donors[train] != train))
        self.assertTrue(set(donors[held].tolist()).issubset(set(train.tolist())))
        self.assertTrue(diagnostic["train_derangement"])
        self.assertTrue(diagnostic["held_donors_train_only"])
        self.assertFalse(diagnostic["outcome_or_target_used"])

    def test_normalizer_uses_only_explicit_fit_matrix(self):
        train = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        first = fit_feature_normalizer(train)
        held = np.asarray([[1e9, -1e9]], dtype=np.float32)
        _ = first.transform(held)
        second = fit_feature_normalizer(train.copy())
        np.testing.assert_array_equal(first.mean, second.mean)
        np.testing.assert_array_equal(first.scale, second.scale)


if __name__ == "__main__":
    unittest.main()
