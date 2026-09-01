import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oracle_headroom", ROOT / "scripts/run_mf3zp_oracle_headroom.py")
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)


class OracleHeadroomTest(unittest.TestCase):
    def test_interval_is_not_collapsed_post_hoc(self):
        self.assertEqual(oracle.premature_bounds(2, (3, 5)), (1, 1))
        self.assertEqual(oracle.premature_bounds(4, (3, 5)), (0, 1))
        self.assertEqual(oracle.premature_bounds(5, (3, 5)), (0, 0))

    def test_fixed_utility(self):
        self.assertAlmostEqual(oracle.utility({"nDTW": .8, "SDTW": .4, "SPL": .2}), .55)


if __name__ == "__main__":
    unittest.main()
