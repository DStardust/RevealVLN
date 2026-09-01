import unittest

from revealnav_mf3.evidence_uad import ConstraintState, derive_constraint_uad, option_readiness
from test_mf3zp_instruction_graph import graph


class EvidenceUADTest(unittest.TestCase):
    def test_k_is_exactly_three(self):
        values = derive_constraint_uad([True]*4, [True]*4, [True]*4)
        self.assertEqual(values, (ConstraintState.A, ConstraintState.A, ConstraintState.D, ConstraintState.D))
        with self.assertRaises(ValueError):
            derive_constraint_uad([True], [True], [True], stability_k=2)

    def test_option_uses_only_decisive_chain(self):
        value = graph()
        self.assertEqual(option_readiness(value, "B1", {"c1":"D", "c2":"A", "c3":"D"}), ConstraintState.A)
        self.assertEqual(option_readiness(value, "B1", {"c1":"U", "c2":"D", "c3":"D"}), ConstraintState.U)
        self.assertEqual(option_readiness(value, "B1", {"c1":"D", "c2":"D", "c3":"D"}), ConstraintState.D)


if __name__ == "__main__":
    unittest.main()
