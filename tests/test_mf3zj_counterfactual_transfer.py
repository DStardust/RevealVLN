from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from revealnav_mf3.uncertainty_gate import (
    CounterfactualTransferGate,
    TRANSFER_FEATURE_NAMES,
    transfer_action_features,
)


class MF3ZJCounterfactualTransferTest(unittest.TestCase):
    def test_compact_features_are_finite_and_source_free(self):
        rng = np.random.default_rng(17)
        embeddings = [rng.normal(size=768).astype(np.float32) for _ in range(4)]
        decision = {
            "step": 4,
            "native_margin": 0.03,
            "current_local_action_ids": ["a", "b", "c"],
            "proposal_source": "must_not_be_read",
        }
        forward = transfer_action_features(decision, *embeddings)
        decision["proposal_source"] = "different_but_irrelevant"
        repeated = transfer_action_features(decision, *embeddings)
        reverse = transfer_action_features(
            decision, embeddings[0], embeddings[1], embeddings[3], embeddings[2]
        )
        self.assertEqual(forward.shape, (len(TRANSFER_FEATURE_NAMES),))
        np.testing.assert_array_equal(forward, repeated)
        self.assertFalse(np.array_equal(forward, reverse))

    def test_gate_requires_nonnegative_return_rule(self):
        dimensions = len(TRANSFER_FEATURE_NAMES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.npz"
            np.savez(
                path,
                means=np.zeros((3, dimensions)),
                scales=np.ones((3, dimensions)),
                return_coefficients=np.zeros((3, dimensions + 1)),
                harm_coefficients=np.zeros((3, dimensions + 1)),
                feature_names=np.asarray(TRANSFER_FEATURE_NAMES),
            )
            with self.assertRaises(RuntimeError):
                CounterfactualTransferGate(
                    path,
                    {
                        "return_threshold": -1e-6,
                        "harm_probability_threshold": 0.6,
                    },
                )
            gate = CounterfactualTransferGate(
                path,
                {
                    "return_threshold": 0.0,
                    "harm_probability_threshold": 0.6,
                },
            )
            self.assertTrue(gate.evaluate(np.zeros(dimensions))["authorized"])


if __name__ == "__main__":
    unittest.main()
