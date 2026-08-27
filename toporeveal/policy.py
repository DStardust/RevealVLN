"""Checkpoint admission and risk-aware sequential branch commitment."""

from __future__ import annotations

from dataclasses import dataclass

from .memory import TopologicalMemory
from .types import (
    BranchCandidate,
    BranchStatus,
    CheckpointProposal,
    Decision,
    DecisionContext,
    DecisionKind,
)


@dataclass(frozen=True)
class PolicyConfig:
    active_option_width: int = 2
    checkpoint_value_threshold: float = 0.25
    minimum_stable_observations: int = 2
    commit_discriminable_threshold: float = 0.70
    commit_evidence_threshold: float = 0.70
    commit_target_threshold: float = 0.60
    minimum_explore_utility: float = 0.05
    maximum_uncommitted_risk: float = 0.35
    minimum_inspect_margin: float = 1.0
    target_weight: float = 1.0
    information_weight: float = 0.75
    constraint_weight: float = 0.50
    travel_cost_weight: float = 0.10
    return_cost_weight: float = 0.10
    irreversible_risk_weight: float = 1.0
    revisit_weight: float = 0.20

    def __post_init__(self) -> None:
        if (
            not isinstance(self.active_option_width, int)
            or isinstance(self.active_option_width, bool)
            or self.active_option_width < 1
        ):
            raise ValueError("active_option_width must be a positive integer")


class CheckpointGate:
    """Admits checkpoints by expected recoverability value, not entropy."""

    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        self.config = config

    def value(self, proposal: CheckpointProposal) -> float:
        return (
            proposal.need_return_probability * proposal.best_alternative_value
            + proposal.irreversible_risk
            * proposal.recovery_cost_without_checkpoint
            - proposal.memory_cost
        )

    def should_create(self, proposal: CheckpointProposal) -> bool:
        return (
            proposal.stable_branch_count >= 2
            and proposal.stable_observations
            >= self.config.minimum_stable_observations
            and self.value(proposal) >= self.config.checkpoint_value_threshold
        )


class BranchPolicy:
    """Chooses between commit, reversible exploration, inspection, and return."""

    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        self.config = config

    def exploration_utility(self, branch: BranchCandidate) -> float:
        config = self.config
        return (
            config.target_weight * branch.target_probability
            + config.information_weight * branch.information_gain
            + config.constraint_weight * branch.constraint_coverage
            - config.travel_cost_weight * branch.travel_cost
            - config.return_cost_weight * branch.return_cost
            - config.irreversible_risk_weight * branch.irreversible_risk
            - config.revisit_weight * branch.visits
        )

    def ranked_viable_options(
        self, memory: TopologicalMemory, checkpoint_id: str
    ) -> tuple[BranchCandidate, ...]:
        """Rank the complete unexhausted local set without deleting its tail."""

        checkpoint = memory.checkpoint(checkpoint_id)
        viable = (
            branch for branch in checkpoint.branches.values()
            if branch.status is BranchStatus.UNTRIED
        )
        return tuple(sorted(
            viable,
            key=lambda branch: (
                -self.exploration_utility(branch),
                -branch.target_probability,
                branch.branch_id,
            ),
        ))

    def active_options(
        self, memory: TopologicalMemory, checkpoint_id: str
    ) -> tuple[BranchCandidate, ...]:
        """Return the dynamic Top-k activity beam; memory retains every branch."""

        return self.ranked_viable_options(
            memory, checkpoint_id
        )[:self.config.active_option_width]

    def decide(
        self, context: DecisionContext, memory: TopologicalMemory
    ) -> Decision:
        if context.goal_found:
            return Decision(DecisionKind.STOP, "goal_or_target_confirmed")

        local = self.active_options(memory, context.current_checkpoint_id)
        target = max(
            local,
            key=lambda branch: (branch.target_probability, branch.branch_id),
            default=None,
        )

        if target is not None and self._commit_ready(context, target):
            return Decision(
                DecisionKind.COMMIT,
                "target_present_and_referential_evidence_closed",
                context.current_checkpoint_id,
                target.branch_id,
            )

        exploratory = max(
            local,
            key=lambda branch: (self.exploration_utility(branch), branch.branch_id),
            default=None,
        )
        if exploratory is not None and self._safe_to_explore(context, exploratory):
            return Decision(
                DecisionKind.EXPLORE,
                "positive_information_value_with_recoverable_motion",
                context.current_checkpoint_id,
                exploratory.branch_id,
            )

        if context.can_inspect and (
            context.last_safe_margin >= self.config.minimum_inspect_margin
        ):
            return Decision(
                DecisionKind.INSPECT,
                "commitment_not_ready_but_safe_observation_remains",
                context.current_checkpoint_id,
            )

        fallback = memory.best_pending_branch(
            self.exploration_utility,
            exclude_checkpoint=context.current_checkpoint_id,
        )
        if fallback is not None:
            checkpoint_id, branch, utility = fallback
            if utility >= self.config.minimum_explore_utility:
                return Decision(
                    DecisionKind.BACKTRACK,
                    "return_to_best_recoverable_untried_option",
                    checkpoint_id,
                    branch.branch_id,
                    memory.shortest_path(context.current_checkpoint_id, checkpoint_id),
                )

        return Decision(
            DecisionKind.FAIL,
            "no_safe_commitment_or_positive_value_recoverable_option",
            context.current_checkpoint_id,
        )

    def _commit_ready(
        self, context: DecisionContext, target: BranchCandidate
    ) -> bool:
        config = self.config
        return (
            context.reveal_belief.discriminable
            >= config.commit_discriminable_threshold
            and context.evidence_complete_probability
            >= config.commit_evidence_threshold
            and target.target_probability >= config.commit_target_threshold
        )

    def _safe_to_explore(
        self, context: DecisionContext, branch: BranchCandidate
    ) -> bool:
        return (
            context.last_safe_margin > 0.0
            and branch.irreversible_risk
            <= self.config.maximum_uncommitted_risk
            and self.exploration_utility(branch)
            >= self.config.minimum_explore_utility
        )


def normalized_entropy(probabilities: list[float]) -> float:
    """Entropy baseline for experiments; not used by the proposed policy."""

    from math import log

    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("probabilities must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    if len(probabilities) == 1:
        return 0.0
    distribution = [probability / total for probability in probabilities]
    return -sum(p * log(p) for p in distribution if p > 0.0) / log(
        len(distribution)
    )
