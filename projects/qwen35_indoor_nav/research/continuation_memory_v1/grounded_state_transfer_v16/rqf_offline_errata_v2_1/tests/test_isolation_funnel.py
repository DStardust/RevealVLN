from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from audit_funnel import _family_record_complete, classify_legacy_result, failure_funnel
from audit_semantics import _trace_event_summary, isolation_audit
from common import digest_bytes
from snapshot_io import SnapshotReader
from helpers import observation, temporary_directory


class EventCompiler:
    def __init__(self, own=False, other=False):
        self.own = own
        self.other = other

    def atoms(self, observations):
        return [
            {"anchor_A": [], "anchor_B": [], "terminal": []},
            {"anchor_A": [1] if self.own else [], "anchor_B": [2] if self.other else [], "terminal": []},
        ]


def trace():
    return {
        "actions": ["L"],
        "observations": [
            observation(0, pixels={"1": 300, "2": 300}),
            observation(1, pixels={"1": 300, "2": 300}),
        ],
        "collisions": 0,
        "complete": False,
    }


def reader_with(root, values):
    records = []
    objects = root / "objects"
    objects.mkdir()
    for path, payload in values.items():
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        sha = digest_bytes(data)
        (objects / sha).write_bytes(data)
        records.append({"kind": "original_v2_input", "source_path": path, "object_sha256": sha})
    return SnapshotReader({"records": records}, objects), {row["source_path"]: row["object_sha256"] for row in records}


class IsolationFunnelTests(unittest.TestCase):
    def test_I01_own_other_joint_states_are_independent(self):
        observed = set()
        for own in (False, True):
            for other in (False, True):
                result = _trace_event_summary(trace(), EventCompiler(own, other), "H_A", "fixture")
                observed.add((result["own_anchor_seen"], result["other_anchor_seen"]))
        self.assertEqual(observed, {(False, False), (False, True), (True, False), (True, True)})

    def test_I02_isolated_partial_history_is_not_family(self):
        weak = {"content_root": "x", "histories": {"H_A": []}, "traces": {"H_A__C0": {}}}
        self.assertFalse(_family_record_complete(weak))

    def test_I03_contamination_without_boundary_is_unresolved(self):
        result = _trace_event_summary(trace(), EventCompiler(True, True), "H_A", "fixture")
        self.assertEqual(result["first_contamination_stage"], "UNRESOLVED")

    def test_I04_partial_family_notice_has_separate_denominator(self):
        with temporary_directory("q35n_i04_") as root:
            history_path = "history.json"
            family_path = "FAMILY.json"
            reader, hashes = reader_with(root, {history_path: trace(), family_path: {"family_id": "partial"}})
            old = {
                "rows": [
                    {"proposal": "p", "yaw": "yaw_00", "history_id": "H_A", "evidence_path": history_path, "evidence_sha256": hashes[history_path]},
                    {"proposal": "p", "yaw": None, "history_id": None, "evidence_path": family_path, "evidence_sha256": hashes[family_path]},
                ]
            }
            proposal = {"id": "p", "role_indices": [0, 1, 2]}
            with patch("audit_semantics.make_compiler", return_value=EventCompiler(True, False)):
                result = isolation_audit(reader, old, [proposal], {"house": "h", "split": "TEST"}, object)
            self.assertEqual(result["history_record_count"], 1)
            self.assertEqual(result["partial_family_notice_count"], 1)

    def test_I05_certified_without_content_root_is_not_accepted(self):
        self.assertFalse(_family_record_complete({"histories": {name: [] for name in ("H_A", "H_B", "H_A_R", "H_B_R")}, "traces": {str(i): {} for i in range(12)}}))

    def test_I06_duplicate_attempts_are_not_overwritten_and_identity_holds(self):
        with temporary_directory("q35n_i06_") as root:
            collection = "research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_repair_rqf_001/COLLECTION_ATTEMPTS.jsonl"
            attempts = "research/continuation_memory_v1/grounded_state_transfer_v16/runs/v16_repair_rqf_001/proposals/p1/REPAIR_YAW_ATTEMPTS.jsonl"
            reader, _ = reader_with(
                root,
                {
                    collection: b'{"status":"START","proposal":"p1"}\n',
                    attempts: b'{"yaw":0,"status":"REJECTED","error":"Rejected(\\"COLLISION\\")"}\n{"yaw":0,"status":"CERTIFIED"}\n',
                },
            )
            result = failure_funnel(reader, [{"id": "p1"}, {"id": "p2"}], {"rows": []})
            self.assertEqual(len(result["identity_conflicts"]), 1)
            self.assertTrue(result["proposal_denominator"]["identity_holds"])
            self.assertTrue(result["yaw_denominator"]["identity_holds"])
            self.assertEqual(result["yaw_denominator"]["N_expected_yaw_slots"], 16)

    def test_I06b_legacy_error_parser_is_conservative(self):
        self.assertEqual(classify_legacy_result("Rejected('COLLISION')"), "TERMINAL_PHYSICAL_REJECTED")
        self.assertEqual(classify_legacy_result("REJECTED ValueError"), "PROGRAM_ERROR")


if __name__ == "__main__":
    unittest.main()
