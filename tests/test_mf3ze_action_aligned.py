from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.action_aligned import (
    FEATURE_NAMES,
    ActionAlignedReturnGate,
    action_aligned_features,
    hierarchical_proposal_tier,
    residual_with_uncertainty_source,
)


class ActionAlignedReturnGateTest(unittest.TestCase):
    def decision(self) -> dict:
        return {
            "step": 2,
            "policy_risk_adjusted_score": 2.3,
            "native_margin": 0.4,
            "minimum_top2_advantage": 1.2,
            "median_top2_advantage": 2.5,
            "robust_top2_advantage": 2.2,
            "ensemble_mad": 0.3,
            "cold_start_floor_ratio": 0.48,
            "cold_start_relative_mad": 0.12,
            "current_local_action_ids": ["a", "b", "c"],
        }

    def test_features_are_finite_and_action_order_sensitive(self):
        rng = np.random.default_rng(7)
        values = [rng.normal(size=768).astype(np.float32) for _ in range(4)]
        forward = action_aligned_features(self.decision(), *values)
        reverse = action_aligned_features(
            self.decision(), values[0], values[1], values[3], values[2]
        )
        self.assertEqual(forward.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(forward).all())
        self.assertFalse(np.array_equal(forward, reverse))

    def test_saved_linear_ensemble_is_loaded_strictly(self):
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
            gate = ActionAlignedReturnGate(
                path,
                {"return_threshold": -0.1, "harm_probability_threshold": 0.6},
            )
            result = gate.evaluate(np.zeros(dimensions))
            self.assertTrue(result["authorized"])
            self.assertEqual(result["robust_expected_utility"], 0.0)
            self.assertEqual(result["upper_harm_probability"], 0.5)

    def test_hierarchical_veto_does_not_consume_other_tier(self):
        common = dict(
            expansion_threshold=1.0,
            core_threshold=2.0,
            upper_threshold=3.0,
            intervened=False,
        )
        self.assertEqual(hierarchical_proposal_tier(
            1.5, core_evaluated=False, expansion_evaluated=False, **common
        ), "expansion")
        self.assertEqual(hierarchical_proposal_tier(
            2.5, core_evaluated=False, expansion_evaluated=True, **common
        ), "core")
        self.assertEqual(hierarchical_proposal_tier(
            1.5, core_evaluated=False, expansion_evaluated=True, **common
        ), None)
        self.assertIsNone(hierarchical_proposal_tier(
            2.5, core_evaluated=False, expansion_evaluated=False,
            **(common | {"intervened": True}),
        ))

    def test_learned_residual_has_priority_over_uncertainty_floor(self):
        self.assertEqual(
            residual_with_uncertainty_source(True, 0.01, 0.05),
            "learned_residual",
        )
        self.assertEqual(
            residual_with_uncertainty_source(False, 0.01, 0.05),
            "uncertainty_floor",
        )
        self.assertIsNone(
            residual_with_uncertainty_source(False, 0.06, 0.05)
        )


if __name__ == "__main__":
    unittest.main()
