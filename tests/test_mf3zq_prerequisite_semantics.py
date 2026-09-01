import unittest

from revealnav_mf3.oracle_reveal_expiry import derive_prerequisite_satisfaction


class PrerequisiteSemanticsTest(unittest.TestCase):
    def test_satisfaction_persists_after_first_complete_prefix(self):
        values = derive_prerequisite_satisfaction(
            {"p": ((False, False, False), (True, True, True), (False, False, False))},
            ("p",),
        )
        self.assertEqual(values["p"], (False, True, True))


if __name__ == "__main__":
    unittest.main()
