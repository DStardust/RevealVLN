"""Instruction-conditioned option memory bound to frozen ETP candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable


class OptionStatus(str, Enum):
    UNTRIED = "UNTRIED"
    ACTIVE = "ACTIVE"
    PRESERVED = "PRESERVED"
    EXHAUSTED = "EXHAUSTED"
    COMMITTED = "COMMITTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class OptionNode:
    option_id: str
    anchor_checkpoint_id: str
    branch_candidate_id: str
    first_seen_step: int
    last_seen_step: int
    required_constraints: tuple[str, ...]
    resolved_constraints: tuple[str, ...]
    unresolved_constraints: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    return_reference: str
    status: OptionStatus

    def __post_init__(self) -> None:
        for value, name in (
            (self.option_id, "option_id"),
            (self.anchor_checkpoint_id, "anchor_checkpoint_id"),
            (self.branch_candidate_id, "branch_candidate_id"),
            (self.return_reference, "return_reference"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for value, name in ((self.first_seen_step, "first_seen_step"), (self.last_seen_step, "last_seen_step")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative integer")
        if self.first_seen_step > self.last_seen_step:
            raise ValueError("option step interval is invalid")
        for values, name in ((self.required_constraints, "required_constraints"), (self.resolved_constraints, "resolved_constraints"), (self.unresolved_constraints, "unresolved_constraints"), (self.evidence_ids, "evidence_ids")):
            values = tuple(str(value) for value in values)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{name} must contain unique nonempty IDs")
        required = set(self.required_constraints)
        if not set(self.resolved_constraints).issubset(required) or not set(self.unresolved_constraints).issubset(required):
            raise ValueError("resolved/unresolved constraints must be required constraints")
        if set(self.resolved_constraints) & set(self.unresolved_constraints):
            raise ValueError("a constraint cannot be both resolved and unresolved")
        status = self.status if isinstance(self.status, OptionStatus) else OptionStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "required_constraints", tuple(self.required_constraints))
        object.__setattr__(self, "resolved_constraints", tuple(self.resolved_constraints))
        object.__setattr__(self, "unresolved_constraints", tuple(self.unresolved_constraints))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


class OptionGraph:
    def __init__(self, nodes: Iterable[OptionNode] = ()) -> None:
        self._nodes: dict[str, OptionNode] = {}
        for node in nodes:
            self.add(node)

    def add(self, node: OptionNode) -> None:
        if node.option_id in self._nodes and self._nodes[node.option_id] != node:
            raise ValueError(f"option identity conflict: {node.option_id}")
        self._nodes[node.option_id] = node

    def get(self, option_id: str) -> OptionNode:
        try:
            return self._nodes[str(option_id)]
        except KeyError as error:
            raise KeyError(f"unknown option: {option_id}") from error

    def nodes(self) -> tuple[OptionNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    def active_options(self) -> tuple[OptionNode, ...]:
        return tuple(node for node in self.nodes() if node.status in (OptionStatus.ACTIVE, OptionStatus.PRESERVED, OptionStatus.UNTRIED))

    def update_seen(self, option_id: str, step: int) -> None:
        node = self.get(option_id)
        if step < node.first_seen_step:
            raise ValueError("option cannot be observed before first_seen_step")
        self._nodes[option_id] = replace(node, last_seen_step=max(node.last_seen_step, int(step)), status=OptionStatus.ACTIVE if node.status is OptionStatus.UNTRIED else node.status)

    def preserve(self, option_id: str) -> None:
        node = self.get(option_id)
        if node.status in (OptionStatus.INVALID, OptionStatus.EXHAUSTED, OptionStatus.COMMITTED):
            raise ValueError("terminal option cannot be preserved")
        self._nodes[option_id] = replace(node, status=OptionStatus.PRESERVED)

    def commit(self, option_id: str, *, readiness: str) -> None:
        node = self.get(option_id)
        if str(getattr(readiness, "value", readiness)) != "D":
            raise ValueError("commit requires hard D readiness")
        if node.status in (OptionStatus.INVALID, OptionStatus.EXHAUSTED):
            raise ValueError("invalid/exhausted option cannot be committed")
        self._nodes[option_id] = replace(node, status=OptionStatus.COMMITTED)

    def exhaust(self, option_id: str) -> None:
        node = self.get(option_id)
        if node.status is OptionStatus.COMMITTED:
            raise ValueError("committed option cannot be exhausted")
        self._nodes[option_id] = replace(node, status=OptionStatus.EXHAUSTED)

    def invalidate(self, option_id: str) -> None:
        self._nodes[option_id] = replace(self.get(option_id), status=OptionStatus.INVALID)

    def snapshot(self) -> tuple[dict[str, object], ...]:
        return tuple({
            "option_id": node.option_id,
            "anchor_checkpoint_id": node.anchor_checkpoint_id,
            "branch_candidate_id": node.branch_candidate_id,
            "first_seen_step": node.first_seen_step,
            "last_seen_step": node.last_seen_step,
            "required_constraints": list(node.required_constraints),
            "resolved_constraints": list(node.resolved_constraints),
            "unresolved_constraints": list(node.unresolved_constraints),
            "evidence_ids": list(node.evidence_ids),
            "return_reference": node.return_reference,
            "status": node.status.value,
        } for node in self.nodes())


__all__ = ["OptionGraph", "OptionNode", "OptionStatus"]
