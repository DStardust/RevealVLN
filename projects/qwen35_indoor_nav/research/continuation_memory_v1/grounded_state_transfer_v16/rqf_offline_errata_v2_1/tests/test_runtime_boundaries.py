from __future__ import annotations

import unittest

from candidate_core import (
    BoundedRunner,
    PhysicalRejected,
    classify_exception,
    preserve_diagnostic_failure,
    run_registered_operations,
    validate_history_isolation,
)
from snapshot_io import EvidenceInvalid, EvidenceMissing
from helpers import FakeBackend


class RuntimeBoundaryTests(unittest.TestCase):
    def test_B01_history_240_allowed_241_blocked_before_step(self):
        backend = FakeBackend()
        runner = BoundedRunner(backend, 240)
        runner.start([0, 0, 0])
        for _ in range(240):
            runner.move("L")
        self.assertEqual(backend.step_calls, 240)
        with self.assertRaisesRegex(PhysicalRejected, "DECISION_BUDGET"):
            runner.move("R")
        self.assertEqual(backend.step_calls, 240)

    def test_B02_full_500th_stop_is_allowed_without_observation(self):
        backend = FakeBackend()
        runner = BoundedRunner(backend, 500)
        runner.start([0, 0, 0])
        for _ in range(499):
            runner.move("L")
        before = backend.observe_calls
        runner.move("S")
        trace = runner.finish(terminal_seen_at_stop=True, event_order_satisfied=True)
        self.assertEqual(len(trace["actions"]), 500)
        self.assertEqual(backend.observe_calls, before)

    def test_B03_501st_decision_is_blocked_before_backend(self):
        backend = FakeBackend()
        runner = BoundedRunner(backend, 500)
        runner.start([0, 0, 0])
        for _ in range(500):
            runner.move("F")
        before = backend.step_calls
        with self.assertRaisesRegex(PhysicalRejected, "DECISION_BUDGET"):
            runner.move("S")
        self.assertEqual(backend.step_calls, before)

    def test_B04_missing_stop_and_action_after_stop_are_rejected(self):
        backend = FakeBackend()
        runner = BoundedRunner(backend, 5)
        runner.start([0, 0, 0])
        runner.move("L")
        with self.assertRaisesRegex(PhysicalRejected, "ACTIVE_STOP_REQUIRED"):
            runner.finish(terminal_seen_at_stop=True, event_order_satisfied=True)
        runner.move("S")
        with self.assertRaisesRegex(PhysicalRejected, "ACTION_AFTER_STOP"):
            runner.move("L")

    def test_B05_collision_rejects_and_stops_control_path(self):
        backend = FakeBackend(collision_at=1)
        runner = BoundedRunner(backend, 5)
        runner.start([0, 0, 0])
        with self.assertRaisesRegex(PhysicalRejected, "COLLISION"):
            runner.move("F")
        self.assertEqual(backend.step_calls, 1)

    def test_B06_history_isolation_requires_own_and_forbids_other(self):
        with self.assertRaises(PhysicalRejected):
            validate_history_isolation(own_anchor_seen=False, other_anchor_seen=False)
        with self.assertRaises(PhysicalRejected):
            validate_history_isolation(own_anchor_seen=True, other_anchor_seen=True)
        validate_history_isolation(own_anchor_seen=True, other_anchor_seen=False)

    def test_B07_stop_requires_terminal_and_event_order(self):
        backend = FakeBackend()
        runner = BoundedRunner(backend, 5)
        runner.start([0, 0, 0])
        runner.move("S")
        with self.assertRaisesRegex(PhysicalRejected, "STOP_TERMINATION_CONDITION"):
            runner.finish(terminal_seen_at_stop=False, event_order_satisfied=True)

    def test_E01_exception_types_and_boundaries_are_distinct(self):
        self.assertEqual(classify_exception(PhysicalRejected("x"), boundary="runtime"), "PHYSICAL_REJECTED")
        self.assertEqual(classify_exception(ValueError("x"), boundary="runtime"), "ERROR")
        self.assertEqual(classify_exception(EvidenceMissing("x"), boundary="frozen_input"), "EVIDENCE_MISSING")
        self.assertEqual(classify_exception(EvidenceInvalid("x"), boundary="frozen_input"), "EVIDENCE_INVALID")
        self.assertEqual(classify_exception(FileNotFoundError("output"), boundary="output"), "ERROR")

    def test_E02_secondary_recording_error_preserves_primary(self):
        def broken_recorder(exc):
            raise OSError("secondary")

        result = preserve_diagnostic_failure(ValueError("primary"), broken_recorder)
        self.assertIn("primary", result["primary_error"])
        self.assertIn("secondary", result["diagnostic_recording_error"])
        self.assertEqual(result["status"], "ERROR")

    def test_E03_interrupt_stops_before_next_operation(self):
        calls = []

        def interrupt():
            calls.append("interrupt")
            raise KeyboardInterrupt()

        def forbidden_next():
            calls.append("next")

        result = run_registered_operations([interrupt, forbidden_next])
        self.assertEqual(result["status"], "INTERRUPTED")
        self.assertEqual(calls, ["interrupt"])


if __name__ == "__main__":
    unittest.main()
