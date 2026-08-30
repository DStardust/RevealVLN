from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.shadow import validate_action_identity


class ActionIdentityTest(unittest.TestCase):
    def test_valid_round_trip(self):
        validate_action_identity(
            ["STOP", "vp-a", "vp-b"],
            [1, 2],
            1,
            2,
            declared_native_id="vp-a",
            declared_adapted_id="vp-b",
            require_non_stop=True,
        )

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            validate_action_identity(
                ["STOP", "vp-a", "vp-a"], [1, 2], 1,
                declared_native_id="vp-a",
            )

    def test_rejects_stop_intervention_and_wrong_feature_id(self):
        with self.assertRaises(ValueError):
            validate_action_identity(
                ["STOP", "vp-a"], [1], 0, 1,
                declared_native_id="STOP",
                declared_adapted_id="vp-a",
                require_non_stop=True,
            )
        with self.assertRaises(ValueError):
            validate_action_identity(
                ["STOP", "vp-a", "vp-b"], [1, 2], 1, 2,
                declared_native_id="vp-b",
                declared_adapted_id="vp-b",
                require_non_stop=True,
            )

    def test_rejects_native_outside_current_candidates(self):
        with self.assertRaises(ValueError):
            validate_action_identity(
                ["STOP", "vp-a", "vp-b"], [2], 1, 2,
                declared_native_id="vp-a",
                declared_adapted_id="vp-b",
                require_non_stop=True,
            )


if __name__ == "__main__":
    unittest.main()
