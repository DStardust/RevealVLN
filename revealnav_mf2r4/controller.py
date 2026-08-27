"""Deterministic online selector for the scoped V4 macro-action pair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence


class BranchMacroAction(str, Enum):
    DEFER = "defer"
    COMMIT = "commit"
    CHECKPOINTED_EXCURSION = "checkpointed_excursion"


@dataclass(frozen=True)
class BranchMacroDecision:
    action: BranchMacroAction
    branch_id: str | None
    predicted_cost: float | None
    preservation_gain: float | None
    reason: str


class BranchExcursionMacroController:
    """Choose between V4 COMMIT(b) and CHECKPOINTED_EXCURSION(b)."""

    def __init__(self, persistence_k: int = 3) -> None:
        if persistence_k < 1:
            raise ValueError("persistence_k must be positive")
        self.persistence_k = persistence_k

    def decide(
        self,
        branch_ids: Sequence[str],
        commit_costs: Sequence[float],
        excursion_costs: Sequence[float],
        stable_observations: int,
    ) -> BranchMacroDecision:
        if len(branch_ids) < 2:
            return BranchMacroDecision(
                BranchMacroAction.DEFER, None, None, None,
                "fewer_than_two_current_branches",
            )
        if not (
            len(branch_ids) == len(commit_costs) == len(excursion_costs)
        ):
            raise ValueError("branch ids and predicted costs must align")
        if len(set(branch_ids)) != len(branch_ids) or any(not x for x in branch_ids):
            raise ValueError("branch ids must be non-empty and unique")
        if stable_observations < self.persistence_k:
            return BranchMacroDecision(
                BranchMacroAction.DEFER, None, None, None,
                "branch_set_not_yet_persistent",
            )
        actions = []
        for branch_id, commit, excursion in zip(
            branch_ids, commit_costs, excursion_costs
        ):
            if not isfinite(commit) or not isfinite(excursion):
                raise ValueError("current branch costs must be finite")
            if commit < 0.0 or excursion < 0.0:
                raise ValueError("current branch costs must be non-negative")
            # Exact ties prefer the less elaborate COMMIT macro.  Branch ids
            # provide a permutation-independent final tie break.
            actions.append((commit, 0, branch_id, BranchMacroAction.COMMIT))
            actions.append((
                excursion, 1, branch_id,
                BranchMacroAction.CHECKPOINTED_EXCURSION,
            ))
        cost, _, branch_id, action = min(actions)
        best_commit = min(commit_costs)
        best_excursion = min(excursion_costs)
        return BranchMacroDecision(
            action=action,
            branch_id=branch_id,
            predicted_cost=float(cost),
            preservation_gain=float(best_commit - best_excursion),
            reason="minimum_locked_v4_predicted_macro_cost",
        )
