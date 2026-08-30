#!/usr/bin/env python3
"""Contracts for V5.21 consensus-gated exploration."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.r2r_consensus_exploration_worker_v5_21 import (
    ConsensusExplorationOptionSearchController,
)


class ConsensusExplorationTests(unittest.TestCase):
    def test_commit_branch_is_the_task_action(self) -> None:
        value = {
            "action": "commit",
            "commit_branch": "native",
            "macro": SimpleNamespace(branch_id="information"),
        }
        selected = ConsensusExplorationOptionSearchController._opp_selected_branch(
            value
        )
        self.assertEqual(selected, "native")

    def test_explore_branch_is_the_task_action(self) -> None:
        value = {
            "action": "explore",
            "commit_branch": None,
            "macro": SimpleNamespace(branch_id="information"),
        }
        selected = ConsensusExplorationOptionSearchController._opp_selected_branch(
            value
        )
        self.assertEqual(selected, "information")

    def test_only_evidence_can_unlock_action_conflict(self) -> None:
        gate = ConsensusExplorationOptionSearchController._gate_origin
        self.assertEqual(
            gate("native", "native", False),
            "consensus_information_probe",
        )
        self.assertEqual(
            gate("replacement", "native", False),
            "native_first_no_direct_replacement",
        )
        self.assertEqual(
            gate("replacement", "native", True),
            "evidence_conditioned_action_update",
        )


if __name__ == "__main__":
    unittest.main()
