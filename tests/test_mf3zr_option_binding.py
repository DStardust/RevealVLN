import unittest

from revealnav_mf3.option_binding_schema import (
    BindingState,
    OptionEvidenceBinding,
    deterministic_option_id,
    reject_forbidden_binding_payload,
)


class Mf3zrOptionBindingTest(unittest.TestCase):
    def _edge(self, **overrides):
        values = {
            "event_id": "e",
            "prefix_step": 2,
            "option_id": "o",
            "candidate_id": "opaque",
            "candidate_rank": 1,
            "constraint_id": "c1",
            "binding_state": BindingState.UNRESOLVED,
            "is_contextual": False,
            "is_discriminative": False,
            "evidence_image_indices": (),
            "evidence_ids": (),
            "source_sha256": "a" * 64,
        }
        values.update(overrides)
        return OptionEvidenceBinding(**values)

    def test_context_and_discriminative_are_exclusive(self):
        with self.assertRaises(ValueError):
            self._edge(is_contextual=True, is_discriminative=True)

    def test_unresolved_machine_edge_is_not_usable(self):
        self.assertFalse(self._edge().usable)
        self.assertTrue(self._edge(verified=True, verification_source="independent_review").usable)

    def test_all_binding_states_are_fixed(self):
        self.assertEqual(
            {item.value for item in BindingState},
            {"SUPPORTS", "CONTRADICTS", "UNRESOLVED", "SHARED_CONTEXT", "NOT_APPLICABLE"},
        )

    def test_option_id_is_deterministic(self):
        self.assertEqual(deterministic_option_id("e", 1, "c"), deterministic_option_id("e", 1, "c"))
        self.assertNotEqual(deterministic_option_id("e", 1, "c"), deterministic_option_id("e", 2, "c"))

    def test_forbidden_payload_fails_closed(self):
        with self.assertRaises(ValueError):
            reject_forbidden_binding_payload({"candidate_id": "x", "delta_utility": 0.1})


if __name__ == "__main__":
    unittest.main()
