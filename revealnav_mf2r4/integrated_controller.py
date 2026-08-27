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


class FrozenOPPEventGate:
    """Project the frozen OPP constraints onto the scoped online overlay."""

    def __init__(
        self, discriminable_threshold: float, evidence_threshold: float,
        target_threshold: float, expiry_threshold: float,
        reveal_threshold: float,
    ) -> None:
        values = (
            discriminable_threshold, evidence_threshold, target_threshold,
            expiry_threshold, reveal_threshold,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("OPP event thresholds must be finite probabilities")
        self.discriminable_threshold = float(discriminable_threshold)
        self.evidence_threshold = float(evidence_threshold)
        self.target_threshold = float(target_threshold)
        self.expiry_threshold = float(expiry_threshold)
        self.reveal_threshold = float(reveal_threshold)

    @staticmethod
    def _probabilities(*values: float) -> None:
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("OPP event beliefs must be finite probabilities")

    def checkpoint_decision(
        self, p_discriminable: float, evidence: float,
        maximum_target_probability: float, reveal_hazard: float,
        expiry_hazard: float,
    ) -> tuple[bool, str]:
        self._probabilities(
            p_discriminable, evidence, maximum_target_probability,
            reveal_hazard, expiry_hazard,
        )
        if (
            p_discriminable >= self.discriminable_threshold
            and evidence >= self.evidence_threshold
            and maximum_target_probability >= self.target_threshold
        ):
            return False, "opp_evidence_ready_base_commit"
        if expiry_hazard >= self.expiry_threshold:
            return True, "opp_expiry_risk_allows_exploration"
        if reveal_hazard >= self.reveal_threshold:
            return False, "opp_reveal_expected_base_inspect"
        return False, "opp_evidence_accumulating_base_follow"

    def initial_action(
        self, p_discriminable: float, evidence: float,
        maximum_target_probability: float, reveal_hazard: float,
        expiry_hazard: float, exploration_available: bool,
    ) -> tuple[str, str]:
        """Return the frozen OPP action class before an outbound move."""

        self._probabilities(
            p_discriminable, evidence, maximum_target_probability,
            reveal_hazard, expiry_hazard,
        )
        if not isinstance(exploration_available, bool):
            raise TypeError("exploration_available must be boolean")
        if (
            p_discriminable >= self.discriminable_threshold
            and evidence >= self.evidence_threshold
            and maximum_target_probability >= self.target_threshold
        ):
            return "commit", "opp_learned_D_and_evidence_closed"
        if expiry_hazard >= self.expiry_threshold:
            if exploration_available:
                return "explore", "opp_last_safe_local_option"
            return "unresolved", "opp_expiry_without_safe_option"
        if reveal_hazard >= self.reveal_threshold:
            return "inspect", "opp_reveal_expected_before_expiry"
        return "follow", "opp_preserve_while_evidence_accumulates"

    def post_excursion_decision(
        self, p_discriminable: float, evidence: float,
        selected_target_probability: float,
    ) -> tuple[bool, str]:
        self._probabilities(
            p_discriminable, evidence, selected_target_probability
        )
        if (
            p_discriminable >= self.discriminable_threshold
            and evidence >= self.evidence_threshold
            and selected_target_probability >= self.target_threshold
        ):
            return True, "opp_selected_branch_discriminable_and_closed"
        return False, "opp_selected_branch_not_safe_to_commit"


class PersistentExcursionLedger:
    """Retain tried-option state across transient controller resets.

    The online harness owns motion execution.  This ledger only authorizes a
    predicted checkpoint benefit above the frozen OPV threshold and records
    the resulting option-state transitions.
    """

    def __init__(self, opv_threshold: float) -> None:
        if not isfinite(opv_threshold) or opv_threshold < 0.0:
            raise ValueError("opv_threshold must be finite and non-negative")
        self.opv_threshold = float(opv_threshold)
        self._statuses: dict[tuple[str, str], OptionStatus] = {}

    @staticmethod
    def _key(checkpoint_id: str, branch_id: str) -> tuple[str, str]:
        if not checkpoint_id or not branch_id:
            raise ValueError("checkpoint and branch ids must be non-empty")
        return checkpoint_id, branch_id

    def register(
        self, checkpoint_id: str, branch_ids: Sequence[str],
    ) -> None:
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch ids must be unique")
        for branch_id in branch_ids:
            self._statuses.setdefault(
                self._key(checkpoint_id, branch_id), OptionStatus.UNTRIED
            )

    def status(self, checkpoint_id: str, branch_id: str) -> OptionStatus:
        return self._statuses.get(
            self._key(checkpoint_id, branch_id), OptionStatus.UNTRIED
        )

    def untried(
        self, checkpoint_id: str, branch_ids: Sequence[str],
    ) -> tuple[str, ...]:
        self.register(checkpoint_id, branch_ids)
        return tuple(
            branch_id for branch_id in branch_ids
            if self.status(checkpoint_id, branch_id) is OptionStatus.UNTRIED
        )

    def authorize(
        self, checkpoint_id: str, decision: BranchMacroDecision,
    ) -> bool:
        if decision.action is not BranchMacroAction.CHECKPOINTED_EXCURSION:
            return False
        if decision.branch_id is None or decision.preservation_gain is None:
            raise ValueError("excursion decision is incomplete")
        return self.authorize_branch(
            checkpoint_id, decision.branch_id, decision.preservation_gain
        )

    def authorize_branch(
        self, checkpoint_id: str, branch_id: str, preservation_gain: float,
    ) -> bool:
        """Activate the branch actually chosen by an unchanged base policy."""

        if not isfinite(preservation_gain):
            raise ValueError("preservation gain must be finite")
        key = self._key(checkpoint_id, branch_id)
        if self._statuses.get(key) is not OptionStatus.UNTRIED:
            return False
        if preservation_gain <= self.opv_threshold:
            return False
        self._statuses[key] = OptionStatus.ACTIVE
        return True

    def resolve_continue(self, checkpoint_id: str, branch_id: str) -> None:
        self._resolve(checkpoint_id, branch_id, OptionStatus.COMMITTED)

    def resolve_return(self, checkpoint_id: str, branch_id: str) -> None:
        self._resolve(checkpoint_id, branch_id, OptionStatus.EXHAUSTED)

    def _resolve(
        self, checkpoint_id: str, branch_id: str, status: OptionStatus,
    ) -> None:
        key = self._key(checkpoint_id, branch_id)
        if self._statuses.get(key) is not OptionStatus.ACTIVE:
            raise RuntimeError("only an active excursion can be resolved")
        self._statuses[key] = status

    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(value is status for value in self._statuses.values())
            for status in OptionStatus
        }


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
