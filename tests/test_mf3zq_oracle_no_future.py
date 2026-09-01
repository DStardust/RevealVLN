import unittest

from revealnav_mf3.oracle_revealskill_schema import reject_forbidden_oracle_mapping


class OracleNoFutureTest(unittest.TestCase):
    def test_forbidden_future_is_rejected(self):
        with self.assertRaises(ValueError):
            reject_forbidden_oracle_mapping({"future_frame": "x"})

    def test_nested_future_is_rejected(self):
        with self.assertRaises(ValueError):
            reject_forbidden_oracle_mapping({"state": {"future_candidate_set": []}})


if __name__ == "__main__":
    unittest.main()
