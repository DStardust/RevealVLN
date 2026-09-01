import unittest

from revealnav_mf3.evidence_option_graph import EvidenceOptionGraph
from revealnav_mf3.option_binding_schema import BindingState, OptionEvidenceBinding


class Mf3zrGraphTest(unittest.TestCase):
    def _edge(self, state, option="o1"):
        return OptionEvidenceBinding(
            event_id="e", prefix_step=0, option_id=option, candidate_id=option,
            candidate_rank=0, constraint_id="c1", binding_state=state,
            is_contextual=state is BindingState.SHARED_CONTEXT,
            is_discriminative=False, evidence_image_indices=(), evidence_ids=(),
            source_sha256="a" * 64,
        )

    def test_shared_context_can_bind_multiple_options(self):
        graph = EvidenceOptionGraph("e", 0, ("c1",), ("o1", "o2"), (
            self._edge(BindingState.SHARED_CONTEXT, "o1"),
            self._edge(BindingState.SHARED_CONTEXT, "o2"),
        ))
        self.assertEqual(len(graph.edges_for_constraint("c1")), 2)

    def test_support_and_contradict_conflict_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceOptionGraph("e", 0, ("c1",), ("o1",), (
                self._edge(BindingState.SUPPORTS), self._edge(BindingState.CONTRADICTS),
            ))


if __name__ == "__main__":
    unittest.main()
