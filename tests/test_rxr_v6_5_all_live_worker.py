#!/usr/bin/env python3

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rxr_v6_5_all_live_worker import (
    AllLiveAlternativeCounterfactualController,
    candidate_set_sha256,
    option_causal_sha256,
    shared_state_sha256,
    trace_macro_observation,
)
from revealnav_mf2r3 import OptionStatus


class V65WorkerIdentityTest(unittest.TestCase):
    def test_candidate_hash_is_order_invariant_and_content_bound(self):
        values = {
            "g2": torch.arange(8, dtype=torch.float32),
            "g0": torch.ones(8),
            "g1": torch.zeros(8),
        }
        left = candidate_set_sha256(values)
        right = candidate_set_sha256(dict(reversed(list(values.items()))))
        self.assertEqual(left, right)
        changed = dict(values)
        changed["g1"] = torch.full((8,), 0.01)
        self.assertNotEqual(left, candidate_set_sha256(changed))

    def test_shared_and_option_hashes_separate_identity(self):
        arrays = {
            "instruction": np.ones(8, dtype=np.float16),
            "native": np.zeros(8, dtype=np.float16),
        }
        shared = shared_state_sha256(arrays, "a" * 64)
        embedding = np.arange(8, dtype=np.float16)
        scalars = np.arange(16, dtype=np.float32)
        left = option_causal_sha256(shared, "g1", embedding, scalars)
        right = option_causal_sha256(shared, "g2", embedding, scalars)
        self.assertNotEqual(left, right)

    def test_only_three_or_four_way_groups_are_eligible(self):
        current = {f"g{x}": object() for x in range(4)}
        age = {f"g{x}": x for x in range(4)}
        controls = AllLiveAlternativeCounterfactualController.live_controls(
            current, "g0", age, lambda _: OptionStatus.UNTRIED,
        )
        self.assertEqual(len(controls), 4)
        self.assertEqual(controls[0], "g0")
        self.assertEqual(
            AllLiveAlternativeCounterfactualController.live_controls(
                {"g0": object(), "g1": object()},
                "g0", {"g0": 0, "g1": 1},
                lambda _: OptionStatus.UNTRIED,
            ),
            (),
        )

    def test_non_untried_options_are_excluded(self):
        current = {f"g{x}": object() for x in range(4)}
        age = {f"g{x}": x for x in range(4)}
        statuses = {"g2": OptionStatus.EXHAUSTED}
        controls = AllLiveAlternativeCounterfactualController.live_controls(
            current, "g0", age,
            lambda branch: statuses.get(branch, OptionStatus.UNTRIED),
        )
        self.assertEqual(controls, ("g0", "g3", "g1"))
        statuses["g1"] = OptionStatus.COMMITTED
        self.assertEqual(
            AllLiveAlternativeCounterfactualController.live_controls(
                current, "g0", age,
                lambda branch: statuses.get(branch, OptionStatus.UNTRIED),
            ),
            (),
        )

    def test_macro_observation_requires_ordered_physical_lifecycle(self):
        target = {
            "decision_index": 0,
            "group_id": "group",
            "option_id": "option",
            "checkpoint_id": "checkpoint",
            "native_branch_id": "native",
            "alternative_branch_id": "alternative",
            "candidate_set_sha256": "c" * 64,
            "shared_state_sha256": "s" * 64,
        }
        group = {
            **target,
            "candidate_branch_ids": ["alternative", "native", "other"],
            "options": [{
                "option_id": "option",
                "option_causal_sha256": "o" * 64,
            }],
        }
        rows = [
            {
                "event": "v6_5_multi_option_intervention",
                "group_id": "group", "native_branch": "native",
                "alternative_branch": "alternative",
                "candidate_set_sha256": "c" * 64,
                "shared_state_sha256": "s" * 64,
            },
            {"event": "return_complete", "success": True},
            {
                "event": "retained_alternative_armed",
                "checkpoint_id": "checkpoint",
                "rejected_native_branch": "native",
                "branch_id": "alternative", "return_verified": True,
            },
            {"event": "checkpoint_topology_restored", "checkpoint_id": "checkpoint"},
            {
                "event": "retained_alternative_committed",
                "checkpoint_id": "checkpoint",
                "rejected_native_branch": "native",
                "branch_id": "alternative",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller_trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            observed = trace_macro_observation(
                {"mode": "macro", "target": target, "candidate_events": [group]},
                Path(directory),
            )
        self.assertTrue(observed["target_physical_return_verified"])
        self.assertTrue(observed["target_topology_restored"])
        self.assertTrue(observed["target_alternative_committed"])
        self.assertEqual(
            observed["committed_alternative_branch_id"], "alternative"
        )


if __name__ == "__main__":
    unittest.main()
