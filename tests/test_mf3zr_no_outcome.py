import unittest

from revealnav_mf3.option_binding_schema import reject_forbidden_binding_payload


class Mf3zrNoOutcomeTest(unittest.TestCase):
    def test_nested_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            reject_forbidden_binding_payload({"evidence": {"route_truth": "x"}})

    def test_future_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            reject_forbidden_binding_payload({"future_frame": "image.jpg"})


if __name__ == "__main__":
    unittest.main()
