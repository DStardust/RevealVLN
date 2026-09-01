import unittest

from revealnav_mf3.option_identity import build_option_identities, candidate_persistence, validate_binding_step


class Mf3zrOptionIdentityTest(unittest.TestCase):
    def test_identity_is_causal_and_birth_checked(self):
        prefixes = [
            {"step": 0, "candidate_ids": ["a", "b"], "source_commitment": "a" * 64},
            {"step": 1, "candidate_ids": ["a"], "source_commitment": "b" * 64},
        ]
        identities, issues = build_option_identities("event", prefixes)
        self.assertFalse(issues)
        a = next(item for item in identities if item.candidate_id == "a")
        validate_binding_step(a, 0)
        with self.assertRaises(ValueError):
            validate_binding_step(a, 2)

    def test_truncated_window_is_unresolved_not_filled(self):
        values = candidate_persistence([
            {"step": 4, "candidate_ids": ["a"], "source_commitment": "a" * 64},
            {"step": 5, "candidate_ids": ["a"], "source_commitment": "b" * 64},
        ])
        self.assertEqual(values[0].persistence_status, "IDENTITY_WINDOW_TRUNCATED")

    def test_gap_is_unresolved(self):
        values = candidate_persistence([
            {"step": 0, "candidate_ids": ["a"], "source_commitment": "a" * 64},
            {"step": 2, "candidate_ids": ["a"], "source_commitment": "b" * 64},
        ])
        self.assertEqual(values[0].persistence_status, "IDENTITY_GAP_UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
