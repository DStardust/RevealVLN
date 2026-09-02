import unittest

from revealnav_mf3.progress_schema import reject_forbidden_progress_payload


class Mf3zvNoOutcomeTest(unittest.TestCase):
    def test_nested_outcome_fields_fail_closed(self):
        for key in ("success", "reward", "spl", "ndtw", "utility", "delta_utility"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reject_forbidden_progress_payload({"safe": {key: 0}})

    def test_safe_causal_fields_pass(self):
        reject_forbidden_progress_payload(
            {"step": 2, "candidate_ids": ["g0", "g1"], "native_action_id": "g1"}
        )


if __name__ == "__main__":
    unittest.main()

