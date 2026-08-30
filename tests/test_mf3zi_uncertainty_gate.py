from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from revealnav_mf3.uncertainty_gate import (
    FEATURE_NAMES,
    UncertaintyReturnGate,
    uncertainty_action_features,
)


class MF3ZIUncertaintyGateTest(unittest.TestCase):
    def decision(self):
        return {
            "step": 3,
            "native_margin": 0.04,
            "current_local_action_ids": ["a", "b", "c"],
        }

    def test_feature_schema_is_finite_and_order_sensitive(self):
        rng = np.random.default_rng(13)
        values = [rng.normal(size=768).astype(np.float32) for _ in range(4)]
        forward = uncertainty_action_features(self.decision(), *values)
        reverse = uncertainty_action_features(
            self.decision(), values[0], values[1], values[3], values[2]
        )
        self.assertEqual(forward.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(forward).all())
        self.assertFalse(np.array_equal(forward, reverse))

    def test_gate_loads_and_applies_fail_closed_rule(self):
        dimensions = len(FEATURE_NAMES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(
                path,
                means=np.zeros((3, dimensions)),
                scales=np.ones((3, dimensions)),
                return_coefficients=np.zeros((3, dimensions + 1)),
                harm_coefficients=np.zeros((3, dimensions + 1)),
                feature_names=np.asarray(FEATURE_NAMES),
            )
            gate = UncertaintyReturnGate(
                path,
                {"return_threshold": -0.1, "harm_probability_threshold": 0.6},
            )
            result = gate.evaluate(np.zeros(dimensions))
            self.assertTrue(result["authorized"])
            self.assertEqual(result["robust_expected_utility"], 0.0)
            self.assertEqual(result["upper_harm_probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
