import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training/mf3zr_option_bound_support_v1"


class Mf3zrSupportArtifactTest(unittest.TestCase):
    def test_fixed_denominator_and_fail_closed_result(self):
        result = json.loads((OUT / "MF3ZR_OPTION_BOUND_SUPPORT_RESULT.json").read_text(encoding="utf-8"))
        self.assertEqual(result["events"], 80)
        self.assertEqual(result["unique_episodes"], 80)
        self.assertEqual(result["joint_supported"], 0)
        self.assertEqual(result["R2R_joint_coverage"], 0.0)
        self.assertEqual(result["RxR_joint_coverage"], 0.0)
        self.assertFalse(result["checkpoint_generated"])
        self.assertEqual(result["oracle_arms_run"], [])
        self.assertFalse(result["public_split_access"]["val_unseen"])

    def test_returnability_has_no_attempts(self):
        audit = json.loads((OUT / "MF3ZR_RETURNABILITY_AUDIT.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["callback_available"])
        self.assertEqual(audit["attempted_count"], 0)
        self.assertEqual(audit["success_count"], 0)


if __name__ == "__main__":
    unittest.main()
