import unittest

from revealnav_mf3.qwen_evidence_annotation import stable_sha256


def causal_prefix(trace, decision_step):
    return stable_sha256([row for row in trace if row["step"] <= decision_step])


class CausalityTest(unittest.TestCase):
    def test_future_and_outcome_mutation_invariance(self):
        trace = [{"step": 0, "candidate_ids": ["B1"]}, {"step": 1, "candidate_ids": ["B2"]}]
        changed_future = [trace[0], {"step": 1, "candidate_ids": ["B9"], "delta_utility": 99}]
        self.assertEqual(causal_prefix(trace, 0), causal_prefix(changed_future, 0))
        self.assertEqual(causal_prefix(trace + [{"step": 2, "outcome": "x"}], 0), causal_prefix(trace, 0))


if __name__ == "__main__":
    unittest.main()
