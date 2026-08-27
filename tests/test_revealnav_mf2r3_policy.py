import unittest

from revealnav_mf2r3 import (
    EvidenceContingentOptionGraph,
    LearnedBranchEstimate,
    LearnedCheckpointGate,
    LearnedOPPConfig,
    LearnedOptionPreservationPolicy,
    OPPAction,
    OPPContext,
    OptionStatus,
    make_ecog_node,
)


def estimate(branch_id, target, q_with, q_without, feasible=True):
    return LearnedBranchEstimate(
        branch_id, target, q_with, q_without, feasible
    )


def context(**overrides):
    values = dict(
        step=4, checkpoint_id="cp-current", stable_observations=3,
        p_unobserved=0.1, p_ambiguous=0.8, p_discriminable=0.1,
        evidence_complete_probability=0.2, reveal_hazard=0.4,
        expiry_hazard=0.2,
        branches=(estimate("b1", .6, 1.0, 2.0),
                  estimate("b2", .4, 1.5, 1.8)),
    )
    values.update(overrides)
    return OPPContext(**values)


class LearnedECOGTest(unittest.TestCase):
    def test_opv_is_exact_q_difference_and_gates_after_k3(self):
        row = estimate("branch", .5, 1.25, 2.0)
        self.assertAlmostEqual(row.opv, .75)
        gate = LearnedCheckpointGate(LearnedOPPConfig(opv_threshold=.5))
        self.assertTrue(gate.should_create(context()))
        self.assertFalse(gate.should_create(context(stable_observations=2)))
        self.assertFalse(gate.should_create(context(branches=(
            estimate("b1", .6, 1.0, 1.1),
            estimate("b2", .4, 1.5, 1.6),
        ))))

    def test_complete_branch_set_and_dynamic_top2(self):
        graph = EvidenceContingentOptionGraph()
        row = context(branches=(
            estimate("b1", .4, 1.0, 2.0),
            estimate("b2", .3, 1.1, 2.0),
            estimate("b3", .2, 1.2, 2.0),
            estimate("b4", .1, 1.3, 2.0),
        ))
        graph.add(make_ecog_node(row, "return/cp", "repr/cp"))
        self.assertEqual(len(graph.node("cp-current").branches), 4)
        self.assertEqual(
            [x.estimate.branch_id for x in graph.active("cp-current")],
            ["b1", "b2"],
        )
        graph.set_status("cp-current", "b1", OptionStatus.EXHAUSTED)
        self.assertEqual(
            [x.estimate.branch_id for x in graph.active("cp-current")],
            ["b2", "b3"],
        )

    def test_never_commits_before_D(self):
        decision = LearnedOptionPreservationPolicy().decide(
            context(), EvidenceContingentOptionGraph()
        )
        self.assertNotEqual(decision.action, OPPAction.COMMIT)

    def test_D_commits_best_target_and_masks_infeasible(self):
        row = context(
            p_unobserved=.05, p_ambiguous=.05, p_discriminable=.9,
            evidence_complete_probability=.9,
            branches=(estimate("bad", .95, .1, .2, False),
                      estimate("good", .8, .5, .8, True)),
        )
        decision = LearnedOptionPreservationPolicy().decide(
            row, EvidenceContingentOptionGraph()
        )
        self.assertEqual(decision.action, OPPAction.COMMIT)
        self.assertEqual(decision.branch_id, "good")

    def test_commit_minimizes_unified_learned_task_cost(self):
        row = context(
            p_unobserved=.05, p_ambiguous=.05, p_discriminable=.9,
            evidence_complete_probability=.9,
            branches=(estimate("high_p_high_route", .9, 4.0, 4.5),
                      estimate("lower_total", .8, .3, .5)),
        )
        decision = LearnedOptionPreservationPolicy().decide(
            row, EvidenceContingentOptionGraph()
        )
        self.assertEqual(decision.action, OPPAction.COMMIT)
        self.assertEqual(decision.branch_id, "lower_total")

    def test_expiry_backtracks_to_saved_untried_option(self):
        graph = EvidenceContingentOptionGraph()
        old = context(step=1, checkpoint_id="cp-old")
        graph.add(make_ecog_node(old, "return/old", "repr/old"))
        decision = LearnedOptionPreservationPolicy().decide(
            context(expiry_hazard=.9), graph
        )
        self.assertEqual(decision.action, OPPAction.BACKTRACK)
        self.assertEqual(decision.checkpoint_id, "cp-old")
        self.assertEqual(decision.branch_id, "b1")

    def test_goal_stops_and_retrieval_is_bounded(self):
        config = LearnedOPPConfig(retrieval_limit=2)
        graph = EvidenceContingentOptionGraph(retrieval_limit=2)
        for index in range(4):
            row = context(step=index, checkpoint_id=f"cp-{index}")
            graph.add(make_ecog_node(row, f"return/{index}", f"repr/{index}"))
        self.assertEqual([node.checkpoint_id for node in graph.retrieve()],
                         ["cp-3", "cp-2"])
        decision = LearnedOptionPreservationPolicy(config).decide(
            context(goal_found=True), graph
        )
        self.assertEqual(decision.action, OPPAction.STOP)


if __name__ == "__main__":
    unittest.main()
