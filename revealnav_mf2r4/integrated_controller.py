"""Closed-loop composition of branch selection, excursion, and return."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from revealnav_mf2r3 import OptionStatus

from .controller import BranchMacroAction, BranchMacroDecision
from .executor import (
    CheckpointReturnExecutor, ExecutorPhase, ReturnCommand,
)
from .fusion import ReeQFusionController


class PostExcursionAction(str, Enum):
    CONTINUE = "continue"
    BACKTRACK = "backtrack"


@dataclass(frozen=True)
class PostExcursionDecision:
    action: PostExcursionAction
    predicted_cost: float
    alternative_cost: float


class StateConditionedReturnExecutor(CheckpointReturnExecutor):
    """V4.5 executor plus the accepted exploration-to-commit transition."""

    def continue_excursion(self) -> None:
        if self.phase is not ExecutorPhase.EXPLORING or self.active_branch is None:
            raise RuntimeError("continue requires an active reached excursion")
        self.branch_status[self.active_branch] = OptionStatus.COMMITTED
        self.phase = ExecutorPhase.COMMITTED


class IntegratedOptionController:
    """Deterministic state machine around independently learned cost heads."""

    def __init__(
        self, checkpoint_id: str, controller_ref: str,
        branch_ids: tuple[str, ...], persistence_k: int = 3,
        wrong_commitment_weight: float = 5.0,
    ) -> None:
        self.fusion = ReeQFusionController(
            persistence_k, wrong_commitment_weight
        )
        self.executor = StateConditionedReturnExecutor(
            checkpoint_id, controller_ref, branch_ids
        )

    def decide_at_checkpoint(
        self, branch_ids: Sequence[str], target_probabilities: Sequence[float],
        commit_costs: Sequence[float], excursion_costs: Sequence[float],
        stable_observations: int,
    ) -> BranchMacroDecision:
        if self.executor.phase is not ExecutorPhase.AT_CHECKPOINT:
            raise RuntimeError("branch selection requires checkpoint state")
        if not (
            len(branch_ids) == len(target_probabilities)
            == len(commit_costs) == len(excursion_costs)
        ):
            raise ValueError("branch estimates must align")
        indices = [
            index for index, branch_id in enumerate(branch_ids)
            if self.executor.branch_status.get(branch_id) is OptionStatus.UNTRIED
        ]
        if not indices:
            raise RuntimeError("checkpoint has no untried branch")
        if len(indices) == 1:
            index = indices[0]
            probability = target_probabilities[index]
            commit = commit_costs[index]
            if (
                not isfinite(probability) or not 0.0 <= probability <= 1.0
                or not isfinite(commit) or commit < 0.0
            ):
                raise ValueError("last branch estimate is invalid")
            branch_id = branch_ids[index]
            decision = BranchMacroDecision(
                BranchMacroAction.COMMIT, branch_id,
                float(commit + self.fusion.wrong_commitment_weight * (1.0 - probability)),
                0.0, "only_untried_persistent_branch",
            )
            self.executor.commit(branch_id)
            return decision
        decision = self.fusion.decide(
            [branch_ids[index] for index in indices],
            [target_probabilities[index] for index in indices],
            [commit_costs[index] for index in indices],
            [excursion_costs[index] for index in indices],
            stable_observations,
        )
        if decision.action is BranchMacroAction.COMMIT:
            self.executor.commit(decision.branch_id)
        elif decision.action is BranchMacroAction.CHECKPOINTED_EXCURSION:
            self.executor.start_excursion(decision.branch_id)
        elif decision.action is not BranchMacroAction.DEFER:
            raise RuntimeError("unsupported initial macro action")
        return decision

    def decide_after_excursion(
        self, continue_cost: float, backtrack_cost: float,
    ) -> tuple[PostExcursionDecision, ReturnCommand | None]:
        if self.executor.phase is not ExecutorPhase.EXPLORING:
            raise RuntimeError("post-excursion decision requires reached excursion")
        if any(
            not isfinite(value) or value < 0
            for value in (continue_cost, backtrack_cost)
        ):
            raise ValueError("post-excursion costs must be finite and non-negative")
        if continue_cost <= backtrack_cost:
            decision = PostExcursionDecision(
                PostExcursionAction.CONTINUE,
                float(continue_cost), float(backtrack_cost),
            )
            self.executor.continue_excursion()
            return decision, None
        decision = PostExcursionDecision(
            PostExcursionAction.BACKTRACK,
            float(backtrack_cost), float(continue_cost),
        )
        return decision, self.executor.request_backtrack()

    def fail_closed_outbound(self) -> ReturnCommand:
        """Return toward the same checkpoint after an outbound control failure."""

        if self.executor.phase is not ExecutorPhase.EXPLORING:
            raise RuntimeError("outbound failure requires an active excursion")
        return self.executor.request_backtrack()

    def report_return(self, succeeded: bool) -> None:
        self.executor.report_return(succeeded)

    def retry_return(self) -> ReturnCommand:
        return self.executor.retry_return()
