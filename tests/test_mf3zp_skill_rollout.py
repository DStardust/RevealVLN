import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skill_rollout", ROOT / "scripts/collect_mf3zp_skill_rollouts.py")
rollout = importlib.util.module_from_spec(spec); spec.loader.exec_module(rollout)


class SkillRolloutTest(unittest.TestCase):
    def test_one_skill_and_frozen_continuation(self):
        row = {"controller_frozen": True, "teleport": False, "public_split": False,
               "high_level_skills": ["FOLLOW", "INSPECT"], "intended_skill_index": 1,
               "changed_skill_indices": [1], "frozen_continuation_sha256": "a", "reference_continuation_sha256": "a"}
        rollout.validate_skill_rollout(row)
        for key, bad in (("teleport", True), ("changed_skill_indices", [0, 1]), ("frozen_continuation_sha256", "b")):
            changed = dict(row); changed[key] = bad
            with self.assertRaises(ValueError): rollout.validate_skill_rollout(changed)


if __name__ == "__main__":
    unittest.main()
