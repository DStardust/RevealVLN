import unittest

from revealnav_mf3.evidence_uad import option_reveal_step, reveal_expiry_slack
from test_mf3zp_instruction_graph import graph


class RevealExpiryTest(unittest.TestCase):
    def test_option_reveal_is_maximum_decisive_constraint_reveal(self):
        self.assertEqual(option_reveal_step(graph(), "B1", {"c1": 1, "c2": 3, "c3": 5}), 5)
        self.assertEqual(reveal_expiry_slack(5, 4), -1)


if __name__ == "__main__":
    unittest.main()
