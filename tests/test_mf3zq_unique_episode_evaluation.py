import unittest

from revealnav_mf3.oracle_headroom_metrics import utility


class UniqueEpisodeEvaluationTest(unittest.TestCase):
    def test_utility_is_fixed(self):
        self.assertAlmostEqual(utility({"nDTW": 1, "SDTW": 0.5, "SPL": 0.25}), 0.6875)


if __name__ == "__main__":
    unittest.main()
