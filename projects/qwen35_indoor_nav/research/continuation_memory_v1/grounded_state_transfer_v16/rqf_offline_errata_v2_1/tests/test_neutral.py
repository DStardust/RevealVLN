from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

from audit_semantics import (
    aggregate_yaws,
    classify_neutral_events,
    classify_probe,
    load_compiler_class,
    make_compiler,
)


PACKAGE = Path(__file__).resolve().parent.parent
V16 = PACKAGE.parent
COMPILER = V16.parents[2] / "data_pipeline/mechanism_factory_v2/compiler.py"


def events(*roles: str):
    hit = set(roles)
    return [
        {"anchor_A": [], "anchor_B": [], "terminal": []},
        {name: ([index + 1] if name in hit else []) for index, name in enumerate(("anchor_A", "anchor_B", "terminal"))},
    ]


def record(yaw: int, status: str):
    return {"yaw": yaw, "neutral_gate_result": status}


class NeutralTests(unittest.TestCase):
    def test_N01_terminal_only_is_neutral(self):
        result = classify_neutral_events(events("terminal"))
        self.assertEqual(result["classification"], "VALID_NEUTRAL_START")
        self.assertTrue(result["terminal_seen"])

    def test_N02_anchor_is_invalid(self):
        self.assertEqual(classify_neutral_events(events("anchor_A"))["classification"], "INVALID_ANCHOR_PRESENT")
        self.assertEqual(classify_neutral_events(events("anchor_B"))["classification"], "INVALID_ANCHOR_PRESENT")

    def test_N03_anchor_and_terminal_rejects_for_anchor(self):
        result = classify_neutral_events(events("anchor_A", "terminal"))
        self.assertEqual(result["anchor_roles_triggered"], ["anchor_A"])
        self.assertTrue(result["terminal_seen"])

    def _compiler(self):
        Compiler = load_compiler_class(COMPILER)
        return Compiler(
            roles={"anchor_A": ["chair", "room"], "anchor_B": ["bed", "room"], "terminal": ["plant", "room"]},
            tasks={"task_A": {"anchor": "anchor_A", "terminal": "terminal", "instruction": "x"}},
            eligible={"anchor_A": [7], "anchor_B": [8], "terminal": [9]},
        )

    def test_N04_see2_255_256_boundary(self):
        observations = [
            {"step": 0, "evidence_complete": True, "pixels": {"7": 256}},
            {"step": 1, "evidence_complete": True, "pixels": {"7": 256}},
            {"step": 2, "evidence_complete": True, "pixels": {"7": 255}},
        ]
        atoms = self._compiler().atoms(observations)
        self.assertEqual(atoms[1]["anchor_A"], [7])
        self.assertEqual(atoms[2]["anchor_A"], [])

    def test_N05_different_instances_do_not_merge(self):
        Compiler = load_compiler_class(COMPILER)
        compiler = Compiler(
            roles={"anchor_A": ["chair", "room"], "anchor_B": ["bed", "room"], "terminal": ["plant", "room"]},
            tasks={"task_A": {"anchor": "anchor_A", "terminal": "terminal", "instruction": "x"}},
            eligible={"anchor_A": [7, 10], "anchor_B": [8], "terminal": [9]},
        )
        observations = [
            {"step": 0, "evidence_complete": True, "pixels": {"7": 300, "10": 0}},
            {"step": 1, "evidence_complete": True, "pixels": {"7": 0, "10": 300}},
        ]
        self.assertEqual(compiler.atoms(observations)[1]["anchor_A"], [])

    def test_N06_nonadjacent_or_unknown_frames_do_not_merge(self):
        observations = [
            {"step": 0, "evidence_complete": True, "pixels": {"7": 300}},
            {"step": 1, "evidence_complete": False, "pixels": {"7": 0}},
            {"step": 2, "evidence_complete": True, "pixels": {"7": 300}},
        ]
        self.assertIsNone(self._compiler().atoms(observations)[2]["anchor_A"])

    def test_N07_incomplete_probe_never_passes(self):
        trace = {"actions": ["L"], "observations": [{"step": 0, "evidence_complete": True, "pixels": {}}], "collisions": 0, "complete": False}
        result = classify_probe(trace, self._compiler(), trace_identity="fixture")
        self.assertNotEqual(result["neutral_gate_result"], "VALID_NEUTRAL_START")

    def test_N08_complete_rollups(self):
        yaws = tuple(range(0, 24, 3))
        all_valid = [record(yaw, "VALID_NEUTRAL_START") for yaw in yaws]
        all_invalid = [record(yaw, "INVALID_ANCHOR_PRESENT") for yaw in yaws]
        mixed = [record(yaw, "VALID_NEUTRAL_START" if index < 4 else "INVALID_ANCHOR_PRESENT") for index, yaw in enumerate(yaws)]
        self.assertEqual(aggregate_yaws(all_valid)["proposal_classification"], "ALL_EIGHT_NEUTRAL_START_VALID")
        self.assertEqual(aggregate_yaws(all_invalid)["proposal_classification"], "ALL_EIGHT_INVALID")
        self.assertEqual(aggregate_yaws(mixed)["proposal_classification"], "MIXED_VALID_INVALID")

    def test_N09_missing_content_is_incomplete(self):
        rows = [record(yaw, "INVALID_ANCHOR_PRESENT") for yaw in range(0, 21, 3)]
        rows.append(record(21, "EVIDENCE_MISSING"))
        self.assertEqual(aggregate_yaws(rows)["proposal_classification"], "INCOMPLETE_EVIDENCE")
        self.assertEqual(aggregate_yaws([record(yaw, "EVIDENCE_MISSING") for yaw in range(0, 24, 3)])["proposal_classification"], "INCOMPLETE_EVIDENCE")

    def test_N10_not_reached_or_error_is_incomplete(self):
        rows = [record(0, "VALID_NEUTRAL_START")] + [record(yaw, "NOT_REACHED") for yaw in range(3, 24, 3)]
        self.assertEqual(aggregate_yaws(rows)["proposal_classification"], "INCOMPLETE_EVIDENCE")
        rows[-1] = record(21, "ERROR")
        self.assertEqual(aggregate_yaws(rows)["proposal_classification"], "INCOMPLETE_EVIDENCE")

    def test_N11_duplicate_and_illegal_yaw_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_YAW"):
            aggregate_yaws([record(0, "VALID_NEUTRAL_START"), record(0, "VALID_NEUTRAL_START")])
        with self.assertRaisesRegex(ValueError, "INVALID_YAW"):
            aggregate_yaws([record(1, "VALID_NEUTRAL_START")])

    def test_N12_real_terminal_only_regression(self):
        source_root = Path(os.environ["Q35N_REAL_SOURCE_ROOT"])
        proposal_id = "V16R1_18fe6790548a53d278cf"
        path = source_root / (
            "projects/qwen35_indoor_nav/research/continuation_memory_v1/grounded_state_transfer_v16/"
            f"runs/v16_repair_rqf_001/proposals/{proposal_id}/yaw_00/INITIAL_VIEW_PROBE.json"
        )
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "05cb829e2c1b383b2478a6859686618f85a9b98bb31098ed83cad9219ff80772")
        proposals = json.loads((V16 / "runs/v16_repair_rqf_001/PROPOSALS_rqfALeAoiTq.json").read_text())
        proposal = next(row for row in proposals if row["id"] == proposal_id)
        manifest = json.loads((V16 / "DATA_MANIFEST.json").read_text())
        house = next(row for row in manifest["houses"] if row["house"] == "rqfALeAoiTq")
        compiler = make_compiler(proposal, house, load_compiler_class(COMPILER))
        result = classify_probe(json.loads(data), compiler, trace_identity=hashlib.sha256(data).hexdigest())
        self.assertEqual(result["neutral_gate_result"], "VALID_NEUTRAL_START")
        self.assertEqual(result["all_roles_triggered"], ["terminal"])


if __name__ == "__main__":
    unittest.main()
