import unittest

from revealnav_mf3.evidence_constraints import ConstraintKind, EvidenceConstraint, InstructionEvidenceGraph


HASH = "0" * 64


def graph():
    return InstructionEvidenceGraph(
        instruction="Pass the sofa, then take the second left doorway.",
        constraints=(
            EvidenceConstraint("c1", ConstraintKind.ENTITY, "sofa", None, None, (), ()),
            EvidenceConstraint("c2", ConstraintKind.TEMPORAL_ORDER, "doorway", "after", "sofa", ("c1",), ("B1",)),
            EvidenceConstraint("c3", ConstraintKind.ORDINAL, "doorway", "ordinal", "2", ("c2",), ("B1",)),
        ),
        parser_model="test", parser_prompt_sha256=HASH,
    )


class InstructionGraphTest(unittest.TestCase):
    def test_dag_frontier_and_option_chain(self):
        value = graph()
        self.assertEqual(value.active_frontier({}), ("c1",))
        self.assertEqual(value.active_frontier({"c1": "D"}), ("c2",))
        self.assertEqual(value.required_for_option("B1"), ("c2", "c3"))

    def test_unknown_dependency_and_cycle_fail(self):
        with self.assertRaises(ValueError):
            InstructionEvidenceGraph("x", (EvidenceConstraint("c1", "ENTITY", "x", None, None, ("missing",), ()),), "m", HASH)
        with self.assertRaises(ValueError):
            InstructionEvidenceGraph("x", (
                EvidenceConstraint("c1", "ENTITY", "x", None, None, ("c2",), ()),
                EvidenceConstraint("c2", "ENTITY", "y", None, None, ("c1",), ()),
            ), "m", HASH)


if __name__ == "__main__":
    unittest.main()
