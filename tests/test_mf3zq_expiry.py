import unittest

from revealnav_mf3.oracle_revealskill_schema import OracleOption
from revealnav_mf3.oracle_reveal_expiry import ReturnabilityOracle


class ExpiryTest(unittest.TestCase):
    def _option(self):
        return OracleOption("o", "g0", "cp0", 0, 0, 0, (), ("c",))

    def test_missing_control_callback_fails_closed(self):
        oracle = ReturnabilityOracle()
        with self.assertRaises(RuntimeError):
            oracle.is_returnable(self._option(), {}, step=0)

    def test_callback_is_control_boundary(self):
        oracle = ReturnabilityOracle(callback=lambda option, state, horizon: horizon == 8)
        self.assertTrue(oracle.is_returnable(self._option(), {}, step=0))


if __name__ == "__main__":
    unittest.main()
