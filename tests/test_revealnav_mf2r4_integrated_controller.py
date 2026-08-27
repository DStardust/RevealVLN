import unittest

from revealnav_mf2r3 import OptionStatus
from revealnav_mf2r4 import (
    BranchMacroAction, ExecutorPhase, IntegratedOptionController,
    PostExcursionAction,
)


class IntegratedOptionControllerTest(unittest.TestCase):
    def controller(self):
        return IntegratedOptionController(
            "cp", "public-controller:cp", ("left", "right")
        )

    def start_excursion(self, controller):
        decision = controller.decide_at_checkpoint(
            ("left", "right"), (0.8, 0.2),
            (2.0, 4.0), (1.0, 3.0), 3,
        )
        self.assertEqual(decision.action, BranchMacroAction.CHECKPOINTED_EXCURSION)
        self.assertEqual(decision.branch_id, "left")

    def test_continue_reaches_committed_state(self):
        controller = self.controller()
        self.start_excursion(controller)
        decision, command = controller.decide_after_excursion(1.0, 2.0)
        self.assertEqual(decision.action, PostExcursionAction.CONTINUE)
        self.assertIsNone(command)
        self.assertEqual(controller.executor.phase, ExecutorPhase.COMMITTED)
        self.assertEqual(controller.executor.branch_status["left"], OptionStatus.COMMITTED)

    def test_backtrack_exhausts_branch_and_filters_it(self):
        controller = self.controller()
        self.start_excursion(controller)
        decision, command = controller.decide_after_excursion(4.0, 1.0)
        self.assertEqual(decision.action, PostExcursionAction.BACKTRACK)
        self.assertEqual(command.checkpoint_id, "cp")
        controller.report_return(True)
        self.assertEqual(controller.executor.branch_status["left"], OptionStatus.EXHAUSTED)
        next_decision = controller.decide_at_checkpoint(
            ("left", "right"), (0.9, 0.1),
            (0.1, 2.0), (0.2, 1.0), 3,
        )
        self.assertEqual(next_decision.branch_id, "right")

    def test_outbound_and_return_failures_are_fail_closed(self):
        controller = self.controller()
        self.start_excursion(controller)
        command = controller.fail_closed_outbound()
        self.assertEqual(command.branch_id, "left")
        controller.report_return(False)
        self.assertEqual(controller.executor.phase, ExecutorPhase.RETURN_FAILED)
        with self.assertRaises(RuntimeError):
            controller.decide_at_checkpoint(
                ("left", "right"), (0.5, 0.5),
                (1.0, 1.0), (1.0, 1.0), 3,
            )
        controller.retry_return()
        controller.report_return(True)
        self.assertEqual(controller.executor.phase, ExecutorPhase.AT_CHECKPOINT)

    def test_tie_prefers_continue_and_invalid_cost_is_rejected(self):
        controller = self.controller()
        self.start_excursion(controller)
        decision, _ = controller.decide_after_excursion(2.0, 2.0)
        self.assertEqual(decision.action, PostExcursionAction.CONTINUE)
        other = self.controller()
        self.start_excursion(other)
        with self.assertRaises(ValueError):
            other.decide_after_excursion(float("nan"), 1.0)


if __name__ == "__main__":
    unittest.main()
