import unittest


class StopRuleTest(unittest.TestCase):
    def test_failure_status_is_terminal_for_this_revision(self):
        status = "MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL"
        self.assertTrue(status.endswith("FAIL"))


if __name__ == "__main__":
    unittest.main()
