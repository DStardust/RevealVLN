import unittest

import torch

from revealnav_mf2r4 import (
    BranchExcursionMacroController, BranchExcursionQHead,
    BranchExcursionQLoss, BranchMacroAction, CheckpointReturnExecutor,
    ExecutorPhase, ReeQFusionController,
)
from revealnav_mf2r3 import OptionStatus
from revealnav_mf2r4.model import BranchExcursionQOutput
from revealnav_mf2r4.stable_losses import StableBranchExcursionQLoss


class BranchExcursionQTest(unittest.TestCase):
    def batch(self):
        torch.manual_seed(7)
        return {
            "history_embeddings": torch.randn(2, 5, 12),
            "candidate_embeddings": torch.randn(2, 5, 3, 12),
            "candidate_mask": torch.tensor([
                [[1, 1, 0]] * 5,
                [[1, 1, 1]] * 5,
            ], dtype=torch.bool),
            "instruction_embedding": torch.randn(2, 12),
            "decision_index": torch.tensor([3, 4]),
            "commit_cost": torch.tensor([[1.0, 6.0, torch.inf], [1.5, 6.5, 7.0]]),
            "excursion_cost": torch.tensor([[1.0, 3.0, torch.inf], [1.5, 3.5, 4.0]]),
        }

    def forward(self, model, batch):
        return model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["decision_index"],
        )

    def test_shapes_loss_and_backward(self):
        batch = self.batch()
        model = BranchExcursionQHead(12, 8, 16.0)
        output = self.forward(model, batch)
        self.assertEqual(output.commit_cost.shape, (2, 3))
        self.assertTrue(torch.isinf(output.commit_cost[0, 2]))
        losses = BranchExcursionQLoss()(output, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_candidate_permutation_equivariance(self):
        batch = self.batch()
        model = BranchExcursionQHead(12, 8, 16.0).eval()
        original = self.forward(model, batch)
        order = torch.tensor([2, 0, 1])
        permuted = dict(batch)
        permuted["candidate_embeddings"] = batch["candidate_embeddings"][:, :, order]
        permuted["candidate_mask"] = batch["candidate_mask"][:, :, order]
        result = self.forward(model, permuted)
        torch.testing.assert_close(result.commit_cost, original.commit_cost[:, order])
        torch.testing.assert_close(
            result.excursion_cost, original.excursion_cost[:, order]
        )

    def test_tie_aware_listwise_loss_rewards_the_optimal_action_set(self):
        batch = self.batch()
        good = BranchExcursionQOutput(
            commit_cost=torch.tensor([[1.0, 5.0, torch.inf], [1.0, 5.0, 6.0]]),
            excursion_cost=torch.tensor([[1.0, 4.0, torch.inf], [1.0, 4.0, 5.0]]),
        )
        bad = BranchExcursionQOutput(
            commit_cost=torch.tensor([[5.0, 1.0, torch.inf], [5.0, 1.0, 1.5]]),
            excursion_cost=torch.tensor([[5.0, 1.0, torch.inf], [5.0, 1.0, 1.5]]),
        )
        objective = StableBranchExcursionQLoss()
        self.assertLess(
            float(objective(good, batch)["listwise"]),
            float(objective(bad, batch)["listwise"]),
        )


class BranchExcursionMacroControllerTest(unittest.TestCase):
    def test_defers_until_branch_set_is_persistent(self):
        controller = BranchExcursionMacroController(3)
        decision = controller.decide(
            ["b", "a"], [2.0, 1.0], [0.5, 3.0], 2
        )
        self.assertEqual(decision.action, BranchMacroAction.DEFER)

    def test_selects_global_minimum_and_is_permutation_invariant(self):
        controller = BranchExcursionMacroController(3)
        first = controller.decide(
            ["b", "a"], [2.0, 1.0], [0.5, 3.0], 3
        )
        second = controller.decide(
            ["a", "b"], [1.0, 2.0], [3.0, 0.5], 3
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first.action, BranchMacroAction.CHECKPOINTED_EXCURSION
        )
        self.assertEqual(first.branch_id, "b")
        self.assertAlmostEqual(first.predicted_cost, 0.5)
        self.assertAlmostEqual(first.preservation_gain, 0.5)

    def test_exact_action_tie_prefers_commit(self):
        decision = BranchExcursionMacroController(3).decide(
            ["b", "a"], [1.0, 2.0], [1.0, 3.0], 3
        )
        self.assertEqual(decision.action, BranchMacroAction.COMMIT)
        self.assertEqual(decision.branch_id, "b")

    def test_fixed_ree_q_fusion_penalizes_unlikely_branch(self):
        decision = ReeQFusionController(3, 5.0).decide(
            ["likely", "cheap"], [0.9, 0.1],
            [1.0, 0.1], [1.2, 0.2], 3,
        )
        self.assertEqual(decision.action, BranchMacroAction.COMMIT)
        self.assertEqual(decision.branch_id, "likely")
        self.assertAlmostEqual(decision.predicted_cost, 1.5)

    def test_fixed_ree_q_fusion_validates_probabilities(self):
        with self.assertRaises(ValueError):
            ReeQFusionController().decide(
                ["a", "b"], [1.1, -0.1], [1.0, 1.0], [1.0, 1.0], 3
            )


class CheckpointReturnExecutorTest(unittest.TestCase):
    def test_excursion_return_then_commit(self):
        executor = CheckpointReturnExecutor(
            "cp-1", "public-controller:cp-1", ("left", "right")
        )
        executor.start_excursion("left")
        self.assertEqual(executor.phase, ExecutorPhase.EXPLORING)
        command = executor.request_backtrack()
        self.assertEqual(command.branch_id, "left")
        self.assertEqual(command.controller_ref, "public-controller:cp-1")
        executor.report_return(True)
        self.assertEqual(executor.phase, ExecutorPhase.AT_CHECKPOINT)
        self.assertEqual(executor.branch_status["left"], OptionStatus.EXHAUSTED)
        executor.commit("right")
        self.assertEqual(executor.phase, ExecutorPhase.COMMITTED)
        self.assertEqual(executor.branch_status["right"], OptionStatus.COMMITTED)

    def test_failed_return_is_fail_closed_and_retryable(self):
        executor = CheckpointReturnExecutor("cp", "controller", ("a", "b"))
        executor.start_excursion("a")
        executor.request_backtrack()
        executor.report_return(False)
        self.assertEqual(executor.phase, ExecutorPhase.RETURN_FAILED)
        with self.assertRaises(RuntimeError):
            executor.commit("b")
        command = executor.retry_return()
        self.assertEqual(command.branch_id, "a")
        executor.report_return(True)
        self.assertEqual(executor.phase, ExecutorPhase.AT_CHECKPOINT)

    def test_executor_does_not_choose_a_branch(self):
        executor = CheckpointReturnExecutor("cp", "controller", ("a", "b"))
        with self.assertRaises(ValueError):
            executor.start_excursion("unknown")


if __name__ == "__main__":
    unittest.main()
