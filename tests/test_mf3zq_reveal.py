import unittest

from revealnav_mf3.oracle_reveal_expiry import reveal_expiry_slack


class RevealTest(unittest.TestCase):
    def test_slack_classes_are_fixed(self):
        self.assertEqual(reveal_expiry_slack(2, 4), ("POSITIVE_SLACK", 2))
        self.assertEqual(reveal_expiry_slack(2, 2), ("TIGHT", 0))
        self.assertEqual(reveal_expiry_slack(4, 2), ("UNRESOLVABLE", -2))


if __name__ == "__main__":
    unittest.main()
