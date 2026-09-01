import unittest

from revealnav_mf3.reveal_expiry_support import (
    ExpirySupportStatus,
    RevealSupportStatus,
    derive_expiry_support,
    reveal_expiry_status,
)


class Mf3zrRevealExpiryTest(unittest.TestCase):
    def test_unavailable_returnability_is_not_imputed(self):
        result = derive_expiry_support("o", [{"from_step": 0, "status": "EXECUTION_UNAVAILABLE"}])
        self.assertEqual(result.status, ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE)

    def test_unknown_slack_is_explicit(self):
        class Stub:
            reveal_step = None
            expiry_step = None
        self.assertEqual(reveal_expiry_status(Stub(), Stub())[0], "UNKNOWN")

    def test_observed_expiry_requires_transition(self):
        result = derive_expiry_support("o", [
            {"from_step": 0, "status": "RETURNABLE"},
            {"from_step": 1, "status": "NOT_RETURNABLE"},
        ])
        self.assertEqual(result.status, ExpirySupportStatus.EXPIRY_OBSERVED)
        self.assertEqual(result.expiry_step, 0)


if __name__ == "__main__":
    unittest.main()
