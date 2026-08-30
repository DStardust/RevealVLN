#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rxr_v6_3_feature_worker import evidence_scalars


class RxrV63EvidenceTest(unittest.TestCase):
    def test_evidence_vector_is_fixed_and_causal(self):
        initial = {
            "p_discriminable": 0.4, "evidence": 0.3,
            "maximum_target_probability": 0.6,
            "reveal_hazard": 0.2, "expiry_hazard": 0.1,
        }
        post = {
            "p_discriminable": 0.5, "evidence": 0.4,
            "selected_target_probability": 0.7,
        }
        value = evidence_scalars(
            {"n": 0.6, "a": 0.4}, "n", "a", initial, 0.2,
            [0.2, -0.1, 0.3], post,
        )
        self.assertEqual(value.shape, (17,))
        self.assertAlmostEqual(float(value[2]), -0.2, places=6)
        self.assertAlmostEqual(float(value[13]), 2.0 / 3.0, places=6)

    def test_missing_branch_is_rejected(self):
        initial = {
            "p_discriminable": 0.4, "evidence": 0.3,
            "maximum_target_probability": 0.6,
            "reveal_hazard": 0.2, "expiry_hazard": 0.1,
        }
        post = {
            "p_discriminable": 0.5, "evidence": 0.4,
            "selected_target_probability": 0.7,
        }
        with self.assertRaisesRegex(ValueError, "selected branch"):
            evidence_scalars(
                {"n": 1.0}, "n", "a", initial, 0.2,
                [0.2, 0.1, 0.3], post,
            )


if __name__ == "__main__":
    unittest.main()
