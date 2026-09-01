import unittest

from revealnav_mf3.oracle_headroom_protocol import PUBLIC_CLOSED


class PublicFailClosedTest(unittest.TestCase):
    def test_all_public_flags_are_false(self):
        self.assertEqual(PUBLIC_CLOSED, {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False})


if __name__ == "__main__":
    unittest.main()
