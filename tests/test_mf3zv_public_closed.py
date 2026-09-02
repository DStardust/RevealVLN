import unittest

from revealnav_mf3.progress_schema import reject_forbidden_progress_payload


class Mf3zvPublicClosedTest(unittest.TestCase):
    def test_any_public_split_field_fails_closed(self):
        for key in ("val_seen", "val_unseen", "test", "test_challenge"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reject_forbidden_progress_payload({key: False})


if __name__ == "__main__":
    unittest.main()

