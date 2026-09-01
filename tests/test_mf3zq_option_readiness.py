import unittest

from revealnav_mf3.oracle_revealskill_schema import (
    OracleReadiness,
    option_readiness,
)


class OptionReadinessTest(unittest.TestCase):
    def test_prerequisite_is_historical_not_k_stable(self):
        self.assertIs(
            option_readiness({"c": OracleReadiness.D}, {"p": True}, ("c",), ("p",)),
            OracleReadiness.D,
        )

    def test_missing_prerequisite_blocks_d(self):
        self.assertIs(
            option_readiness({"c": OracleReadiness.D}, {"p": False}, ("c",), ("p",)),
            OracleReadiness.A,
        )

    def test_unobserved_dec_is_u(self):
        self.assertIs(
            option_readiness({"c": OracleReadiness.U}, {}, ("c",), ()),
            OracleReadiness.U,
        )


if __name__ == "__main__":
    unittest.main()
