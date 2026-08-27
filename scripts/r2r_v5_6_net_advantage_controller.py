"""Conservative online Net-Advantage veto for the validated V5.6 controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import r2r_action_enabled_pilot_worker_v5 as pilot
import r2r_full_opp_worker_v5_6 as v56
from revealnav_net_advantage import OnlineNetAdvantageScorer


BASE_V56_CONTROLLER = v56.FullOPPActionController


class V56NetAdvantageController(BASE_V56_CONTROLLER):
    """Keep V5.6's proposal only when the causal learned veto approves it."""

    def __init__(
        self, seed: int, mode: str, device: torch.device, trace_path: Path,
        net_advantage_checkpoint: Path, expected_checkpoint_seed: int | None = None,
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
            net_advantage_checkpoint, device, require_online_threshold=True,
            expected_seed=expected_checkpoint_seed,
        )
        self.net_advantage_approvals = 0
        self.net_advantage_vetoes = 0
        self.net_advantage_decisions = 0
        self._native_branch: str | None = None

    @staticmethod
    def _distance(left, right) -> float:
        return float(np.linalg.norm(
            np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
        ))

    def _score_alternatives(self, current, persistent, native):
        graph = pilot._TRAINER.gmaps[0]
        checkpoint_id = pilot._CURRENT_IDS[0]
        if (
            native not in current
            or checkpoint_id not in graph.node_pos
            or native not in graph.ghost_aug_pos
        ):
            return None
        alternatives = {
            branch: current[branch] for branch in persistent
            if (
                branch != native and branch in current
                and branch in graph.ghost_aug_pos
            )
        }
        if not alternatives:
            return None
        checkpoint = graph.node_pos[checkpoint_id]
        return self.net_advantage.score_candidates(
            self.instruction, self.latest_history,
            torch.stack([row[0] for row in self.rows]).mean(0), current[native],
            alternatives,
            self._distance(checkpoint, graph.ghost_aug_pos[native]),
            {
                branch: self._distance(checkpoint, graph.ghost_aug_pos[branch])
                for branch in alternatives
            },
        )

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
        rows = self._score_alternatives(current, persistent, native)
        self.net_advantage_decisions += 1
        if rows is None or proposed not in {row["branch_id"] for row in rows}:
            self.net_advantage_vetoes += 1
            return {**value, "action": "follow", "reason": "net_advantage_missing_causal_input"}
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


class NetAdvantageOnlyController(V56NetAdvantageController):
    """Ablate V5.6 proposals and directly use the sparse causal ranker."""

    def _initial_decision(self, current, persistent, native_branch):
        self._native_branch = native_branch
        try:
            self.rows.append((self.latest_history, current))
            rows = self._score_alternatives(current, persistent, native_branch)
            self.net_advantage_decisions += 1
            if rows is None:
                self.net_advantage_vetoes += 1
                self.follow_delegations += 1
                self._reset_search()
                return None
            proposed = max(rows, key=lambda row: (
                row["net_advantage_score_m"], row["branch_id"]
            ))["branch_id"]
            approved = self.net_advantage.approve(proposed, rows)
            self.record(
                "net_advantage_only_decision", proposed_branch=proposed,
                native_branch=native_branch, approved=approved,
                calibrated_threshold_m=self.net_advantage.threshold,
                candidates=rows,
            )
            self._reset_search()
            if not approved:
                self.net_advantage_vetoes += 1
                self.follow_delegations += 1
                return None
            self.net_advantage_approvals += 1
            self.commit_decisions += 1
            self.effective_commit_interventions += int(
                proposed != native_branch
            )
            return None if self.mode == "shadow" else proposed
        finally:
            self._native_branch = None


class V56NetAdvantageNoReturnController(V56NetAdvantageController):
    """Ablate ECOG trials while keeping the same V5.6 proposal and veto."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_return_suppressions = 0

    def _evaluate(self, current, persistent):
        value = super()._evaluate(current, persistent)
        if value is not None and value["action"] == "explore":
            self.no_return_suppressions += 1
            self.record(
                "ecog_trial_ablation_suppression",
                proposed_branch=value["macro"].branch_id,
                reason="V5.13.1_no_return_ablation",
            )
            return {
                **value, "action": "follow",
                "reason": "V5.13.1_no_return_ablation",
            }
        return value
