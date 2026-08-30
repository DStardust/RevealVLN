from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.exact_replay import (
    ProposalEventIdentity,
    validate_collection_scope,
    validate_exact_prefix,
    validate_forced_switch,
    validate_shadow_event,
)


class ExactReplayInvariantTest(unittest.TestCase):
    def setUp(self):
        self.core = ProposalEventIdentity(
            "RxR", "17", "17DRP5sb8fy", 3, "core", "g1", "g2"
        )
        self.expansion = ProposalEventIdentity(
            "RxR", "17", "17DRP5sb8fy", 7, "expansion", "g4", "g5"
        )

    def test_native_shadow_never_changes_action(self):
        record = {
            "mode": "native_shadow", "action_changed": False,
            "native_action_id": "g1", "adapted_action_id": "g1",
            "event_identity": self.core.__dict__,
        }
        self.assertEqual(validate_shadow_event(record), self.core)
        record["action_changed"] = True
        with self.assertRaises(ValueError):
            validate_shadow_event(record)

    def test_core_and_later_expansion_have_distinct_identity(self):
        self.assertNotEqual(self.core, self.expansion)
        self.assertEqual(len({self.core, self.expansion}), 2)

    def test_targeted_treatment_changes_only_declared_event(self):
        records = [
            {"step": 0, "action_changed": False, "event_identity": None},
            {
                "step": 3, "action_changed": True,
                "native_action_id": "g1", "adapted_action_id": "g2",
                "event_identity": self.core.__dict__,
            },
            {"step": 4, "action_changed": False, "event_identity": None},
        ]
        validate_forced_switch(records, self.core)
        records[-1]["action_changed"] = True
        with self.assertRaises(ValueError):
            validate_forced_switch(records, self.core)

    def test_exact_prefix_checks_physical_trace(self):
        native = [
            {"act": 4, "ghost_vp": "a", "cur_vp": "x", "front_vp": "y", "back_path_len": 0},
            {"act": 4, "ghost_vp": "b", "cur_vp": "y", "front_vp": "z", "back_path_len": 1},
        ]
        validate_exact_prefix(native, [dict(row) for row in native], 1)
        changed = [dict(row) for row in native]
        changed[0]["ghost_vp"] = "wrong"
        with self.assertRaises(ValueError):
            validate_exact_prefix(native, changed, 1)

    def test_scope_rejects_consumed_or_public_data(self):
        allowed = {"17DRP5sb8fy"}
        validate_collection_scope(
            dataset="RxR", split="train", scene_id="17DRP5sb8fy",
            allowed_scenes=allowed, consumed_scenes=set(),
        )
        with self.assertRaises(ValueError):
            validate_collection_scope(
                dataset="RxR", split="val_unseen", scene_id="17DRP5sb8fy",
                allowed_scenes=allowed, consumed_scenes=set(),
            )
        with self.assertRaises(ValueError):
            validate_collection_scope(
                dataset="RxR", split="train", scene_id="17DRP5sb8fy",
                allowed_scenes=allowed, consumed_scenes=allowed,
            )


if __name__ == "__main__":
    unittest.main()
