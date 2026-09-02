import unittest

from revealnav_mf3.mf3zv_protocol import GATES, PUBLIC_CLOSED, REVISION, STATUS_SEALED, validate_protocol


class Mf3zvNoTrainingTest(unittest.TestCase):
    def protocol(self):
        return {
            "revision": REVISION,
            "status": STATUS_SEALED,
            "progress_families": ["ORDINAL", "PASSED_LANDMARK"],
            "review": {"maximum": 100},
            "gates": GATES,
            "public_split_access": PUBLIC_CLOSED,
            "training_run": False,
            "navigation_run": False,
            "checkpoint_generated": False,
        }

    def test_support_protocol_has_no_training(self):
        validate_protocol(self.protocol())

    def test_training_true_fails(self):
        protocol = self.protocol()
        protocol["training_run"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()

