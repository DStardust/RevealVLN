"""Learned Evidence-Contingent Option Graph and preservation policy.

This module consumes learned REE/Q outputs.  It never accepts entropy,
simulator pose, navmesh state, future observations, or rollout labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class OptionStatus(str, Enum):
    UNTRIED = "untried"
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    COMMITTED = "committed"


class OPPAction(str, Enum):
    FOLLOW = "follow"
    INSPECT = "inspect"
    EXPLORE = "explore"
    BACKTRACK = "backtrack"
    COMMIT = "commit"
    STOP = "stop"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class LearnedBranchEstimate:
    branch_id: str
    target_probability: float
    q_with_checkpoint: float
    q_without_checkpoint: float
    feasible: bool

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must be non-empty")
        if not 0.0 <= self.target_probability <= 1.0:
            raise ValueError("target_probability must be in [0, 1]")
        for name in ("q_with_checkpoint", "q_without_checkpoint"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.q_with_checkpoint > self.q_without_checkpoint + 1e-6:
            raise ValueError("Q_with cannot exceed Q_without")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be boolean")

    @property
    def opv(self) -> float:
        """Option value is derived only from the paired learned Q values."""

        return max(0.0, self.q_without_checkpoint - self.q_with_checkpoint)


@dataclass
class ECOGBranch:
    estimate: LearnedBranchEstimate
    status: OptionStatus = OptionStatus.UNTRIED


@dataclass
class ECOGNode:
    checkpoint_id: str
    controller_ref: str
    representative_ref: str
    unresolved_evidence: tuple[str, ...]
    reveal_hazard: float
    expiry_hazard: float
    created_step: int
    branches: dict[str, ECOGBranch] = field(default_factory=dict)


class EvidenceContingentOptionGraph:
    """Sparse graph retaining complete branch sets at valuable checkpoints."""

    def __init__(self, retrieval_limit: int = 8, active_width: int = 2) -> None:
        if retrieval_limit < 1 or active_width < 1:
            raise ValueError("retrieval_limit and active_width must be positive")
        self.retrieval_limit = retrieval_limit
        self.active_width = active_width
        self._nodes: dict[str, ECOGNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def node(self, checkpoint_id: str) -> ECOGNode:
        try:
            return self._nodes[checkpoint_id]
        except KeyError as error:
            raise KeyError(f"unknown ECOG checkpoint: {checkpoint_id}") from error

    def add(self, node: ECOGNode) -> None:
        if not node.checkpoint_id or node.checkpoint_id in self._nodes:
            raise ValueError("checkpoint id must be non-empty and unique")
        if not node.controller_ref or not node.representative_ref:
            raise ValueError("ECOG nodes require return and representation refs")
        if len(node.branches) < 2:
            raise ValueError("ECOG checkpoint must retain at least two branches")
        if set(node.branches) != {
            branch.estimate.branch_id for branch in node.branches.values()
        }:
            raise ValueError("ECOG branch map is inconsistent")
        self._nodes[node.checkpoint_id] = node

    def set_status(
        self, checkpoint_id: str, branch_id: str, status: OptionStatus,
    ) -> None:
        if not isinstance(status, OptionStatus):
            raise TypeError("status must be OptionStatus")
        self.node(checkpoint_id).branches[branch_id].status = status

    @staticmethod
    def _rank(branch: ECOGBranch) -> tuple[float, float, str]:
        estimate = branch.estimate
        return (
            estimate.q_with_checkpoint,
            -estimate.target_probability,
            estimate.branch_id,
        )

    def viable(self, checkpoint_id: str) -> tuple[ECOGBranch, ...]:
        return tuple(sorted(
            (
                branch for branch in self.node(checkpoint_id).branches.values()
                if branch.status is OptionStatus.UNTRIED
                and branch.estimate.feasible
            ),
            key=self._rank,
        ))

    def active(self, checkpoint_id: str) -> tuple[ECOGBranch, ...]:
        """Dynamic top-2 view; lower-ranked branches remain stored."""

        return self.viable(checkpoint_id)[:self.active_width]

    def retrieve(self) -> tuple[ECOGNode, ...]:
        """Return no more than M recent option-bearing nodes."""

        candidates = (
            node for node in self._nodes.values()
            if any(
                branch.status is OptionStatus.UNTRIED
                and branch.estimate.feasible
                for branch in node.branches.values()
            )
        )
        return tuple(sorted(
            candidates,
            key=lambda node: (-node.created_step, node.checkpoint_id),
        )[:self.retrieval_limit])

    def best_fallback(
        self, exclude_checkpoint: str | None = None,
        score=None,
    ) -> tuple[ECOGNode, ECOGBranch] | None:
        if score is None:
            score = lambda estimate: estimate.q_with_checkpoint
        choices = [
            (score(branch.estimate), node.created_step,
             node.checkpoint_id, branch.estimate.branch_id, node, branch)
            for node in self.retrieve()
            if node.checkpoint_id != exclude_checkpoint
            for branch in self.viable(node.checkpoint_id)
        ]
        if not choices:
            return None
        *_, node, branch = min(choices, key=lambda row: row[:4])
        return node, branch


@dataclass(frozen=True)
class OPPContext:
    step: int
    checkpoint_id: str
    stable_observations: int
    p_unobserved: float
    p_ambiguous: float
    p_discriminable: float
    evidence_complete_probability: float
    reveal_hazard: float
    expiry_hazard: float
    branches: tuple[LearnedBranchEstimate, ...]
    can_follow: bool = True
    can_inspect: bool = True
    goal_found: bool = False

    def __post_init__(self) -> None:
        probabilities = (
            self.p_unobserved, self.p_ambiguous, self.p_discriminable
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("U/A/D probabilities must be in [0, 1]")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError("U/A/D probabilities must sum to one")
        for name in (
            "evidence_complete_probability", "reveal_hazard", "expiry_hazard"
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.step < 0 or self.stable_observations < 0:
            raise ValueError("step and stable_observations must be non-negative")
        ids = [branch.branch_id for branch in self.branches]
        if len(ids) != len(set(ids)):
            raise ValueError("branch ids must be unique")

    @property
    def opv(self) -> float:
        return max((branch.opv for branch in self.branches), default=0.0)


@dataclass(frozen=True)
class OPPDecision:
    action: OPPAction
    reason: str
    checkpoint_id: str | None = None
    branch_id: str | None = None
    predicted_cost: float | None = None


@dataclass(frozen=True)
class LearnedOPPConfig:
    persistence_k: int = 3
    opv_threshold: float = 0.10
    discriminable_threshold: float = 0.50
    evidence_threshold: float = 0.50
    target_threshold: float = 0.40
    expiry_threshold: float = 0.50
    reveal_threshold: float = 0.50
    active_width: int = 2
    retrieval_limit: int = 8
    wrong_commitment_weight: float = 5.0


class LearnedCheckpointGate:
    def __init__(self, config: LearnedOPPConfig = LearnedOPPConfig()) -> None:
        self.config = config

    def should_create(self, context: OPPContext) -> bool:
        return (
            len(context.branches) >= 2
            and context.stable_observations >= self.config.persistence_k
            and context.opv > self.config.opv_threshold
            and any(branch.feasible for branch in context.branches)
        )


class LearnedOptionPreservationPolicy:
    """Constrained minimum learned-cost policy over safe actions."""

    def __init__(self, config: LearnedOPPConfig = LearnedOPPConfig()) -> None:
        self.config = config

    def commit_cost(self, branch: LearnedBranchEstimate) -> float:
        return (
            branch.q_without_checkpoint
            + self.config.wrong_commitment_weight
            * (1.0 - branch.target_probability)
        )

    def preserved_cost(self, branch: LearnedBranchEstimate) -> float:
        return (
            branch.q_with_checkpoint
            + self.config.wrong_commitment_weight
            * (1.0 - branch.target_probability)
        )

    def _fallback(self, graph, exclude_checkpoint):
        return graph.best_fallback(exclude_checkpoint, self.preserved_cost)

    @staticmethod
    def _viable(context: OPPContext) -> tuple[LearnedBranchEstimate, ...]:
        return tuple(sorted(
            (branch for branch in context.branches if branch.feasible),
            key=lambda branch: (
                branch.q_with_checkpoint,
                -branch.target_probability,
                branch.branch_id,
            ),
        ))

    def decide(
        self, context: OPPContext, graph: EvidenceContingentOptionGraph,
    ) -> OPPDecision:
        if context.goal_found:
            return OPPDecision(OPPAction.STOP, "goal_or_object_confirmed")
        viable = self._viable(context)
        commit_candidates = tuple(
            branch for branch in viable
            if branch.target_probability >= self.config.target_threshold
        )
        target = min(
            commit_candidates,
            key=lambda branch: (self.commit_cost(branch), branch.branch_id),
            default=None,
        )
        if (
            target is not None
            and context.p_discriminable >= self.config.discriminable_threshold
            and context.evidence_complete_probability >= self.config.evidence_threshold
        ):
            fallback = self._fallback(graph, context.checkpoint_id)
            if fallback is not None:
                node, branch = fallback
                fallback_cost = self.preserved_cost(branch.estimate)
                if fallback_cost + 1e-6 < self.commit_cost(target):
                    return OPPDecision(
                        OPPAction.BACKTRACK,
                        "saved_option_has_lower_predicted_task_loss",
                        node.checkpoint_id, branch.estimate.branch_id,
                        fallback_cost,
                    )
            return OPPDecision(
                OPPAction.COMMIT, "learned_D_and_evidence_closed",
                context.checkpoint_id, target.branch_id,
                self.commit_cost(target),
            )

        if context.expiry_hazard >= self.config.expiry_threshold:
            fallback = self._fallback(graph, context.checkpoint_id)
            if fallback is not None:
                node, branch = fallback
                return OPPDecision(
                    OPPAction.BACKTRACK, "expiry_risk_prefers_saved_option",
                    node.checkpoint_id, branch.estimate.branch_id,
                    self.preserved_cost(branch.estimate),
                )
            if viable:
                branch = viable[0]
                return OPPDecision(
                    OPPAction.EXPLORE, "last_safe_local_option",
                    context.checkpoint_id, branch.branch_id,
                    branch.q_with_checkpoint,
                )
            return OPPDecision(
                OPPAction.UNRESOLVED, "expiry_without_safe_option",
                context.checkpoint_id,
            )

        if (
            context.can_inspect
            and context.reveal_hazard >= self.config.reveal_threshold
        ):
            return OPPDecision(
                OPPAction.INSPECT, "reveal_expected_before_expiry",
                context.checkpoint_id,
            )
        if context.can_follow:
            return OPPDecision(
                OPPAction.FOLLOW, "preserve_option_while_evidence_accumulates",
                context.checkpoint_id,
            )
        if viable:
            branch = viable[0]
            return OPPDecision(
                OPPAction.EXPLORE, "minimum_predicted_safe_cost",
                context.checkpoint_id, branch.branch_id,
                branch.q_with_checkpoint,
            )
        fallback = self._fallback(graph, context.checkpoint_id)
        if fallback is not None:
            node, branch = fallback
            return OPPDecision(
                OPPAction.BACKTRACK, "local_options_exhausted",
                node.checkpoint_id, branch.estimate.branch_id,
                self.preserved_cost(branch.estimate),
            )
        return OPPDecision(
            OPPAction.UNRESOLVED, "no_safe_action", context.checkpoint_id,
        )


def make_ecog_node(
    context: OPPContext, controller_ref: str, representative_ref: str,
    unresolved_evidence: tuple[str, ...] = (),
) -> ECOGNode:
    return ECOGNode(
        checkpoint_id=context.checkpoint_id,
        controller_ref=controller_ref,
        representative_ref=representative_ref,
        unresolved_evidence=unresolved_evidence,
        reveal_hazard=context.reveal_hazard,
        expiry_hazard=context.expiry_hazard,
        created_step=context.step,
        branches={
            branch.branch_id: ECOGBranch(branch) for branch in context.branches
        },
    )
