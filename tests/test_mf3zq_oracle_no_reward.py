import unittest

from revealnav_mf3.oracle_revealskill_schema import reject_forbidden_oracle_mapping


class OracleNoRewardTest(unittest.TestCase):
    def test_reward_and_outcome_are_rejected(self):
        for key in ("reward", "delta_utility", "success", "car_result"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reject_forbidden_oracle_mapping({key: 0})


if __name__ == "__main__":
    unittest.main()
