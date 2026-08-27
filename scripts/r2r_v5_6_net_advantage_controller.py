"""Conservative online Net-Advantage veto for the validated V5.6 controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import r2r_action_enabled_pilot_worker_v5 as pilot
import r2r_full_opp_worker_v5_6 as v56
from revealnav_net_advantage import OnlineNetAdvantageScorer


class V56NetAdvantageController(v56.FullOPPActionController):
    """Keep V5.6's proposal only when the causal learned veto approves it."""

    def __init__(
        self, seed: int, mode: str, device: torch.device, trace_path: Path,
        net_advantage_checkpoint: Path,
    ) -> None:
        super().__init__(seed, mode, device, trace_path)
        net_advantage_checkpoint = net_advantage_checkpoint.resolve()
        if (
            v56.ROOT not in net_advantage_checkpoint.parents
            or net_advantage_checkpoint.is_symlink()
            or not net_advantage_checkpoint.is_file()
        ):
            raise RuntimeError("net-advantage checkpoint must be a project-local file")
        self.net_advantage = OnlineNetAdvantageScorer.from_checkpoint(
            net_advantage_checkpoint, device, require_online_threshold=True
        )
        self.net_advantage_approvals = 0
        self.net_advantage_vetoes = 0
        self._native_branch: str | None = None

    @staticmethod
    def _distance(left, right) -> float:
        return float(np.linalg.norm(
            np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
        ))

    def _evaluate(self, current, persistent):
        value = super()._evaluate(current, persistent)
        if value is None or value["action"] not in ("commit", "explore"):
            return value
        proposed = (
            value["commit_branch"] if value["action"] == "commit"
            else value["macro"].branch_id
        )
        native = self._native_branch
        if proposed == native:
            return value
        graph = pilot._TRAINER.gmaps[0]
        checkpoint_id = pilot._CURRENT_IDS[0]
        if (
            native not in current or proposed not in current
            or checkpoint_id not in graph.node_pos
            or native not in graph.ghost_aug_pos
        ):
            self.net_advantage_vetoes += 1
            return {**value, "action": "follow", "reason": "net_advantage_missing_causal_input"}
        alternatives = {
            branch: current[branch] for branch in persistent
            if branch != native and branch in graph.ghost_aug_pos
        }
        if proposed not in alternatives:
            self.net_advantage_vetoes += 1
            return {**value, "action": "follow", "reason": "net_advantage_proposal_not_scored"}
        checkpoint = graph.node_pos[checkpoint_id]
        rows = self.net_advantage.score_candidates(
            self.instruction, self.latest_history,
            torch.stack([row[0] for row in self.rows]).mean(0), current[native],
            alternatives,
            self._distance(checkpoint, graph.ghost_aug_pos[native]),
            {
                branch: self._distance(checkpoint, graph.ghost_aug_pos[branch])
                for branch in alternatives
            },
        )
        approved = self.net_advantage.approve(proposed, rows)
        self.record(
            "net_advantage_veto", proposed_branch=proposed,
            native_branch=native, approved=approved,
            calibrated_threshold_m=self.net_advantage.threshold,
            candidates=rows,
        )
        if approved:
            self.net_advantage_approvals += 1
            return value
        self.net_advantage_vetoes += 1
        return {**value, "action": "follow", "reason": "net_advantage_veto"}

    def _initial_decision(self, current, persistent, native_branch):
        self._native_branch = native_branch
        try:
            return super()._initial_decision(current, persistent, native_branch)
        finally:
            self._native_branch = None
