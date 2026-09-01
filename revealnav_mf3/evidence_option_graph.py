"""Serializable bipartite evidence--option graph for MF3ZR.

The graph is a bookkeeping representation.  It permits shared context edges
and does not force a one-hot assignment of a constraint to one candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .option_binding_schema import BindingState, OptionEvidenceBinding


@dataclass(frozen=True)
class EvidenceOptionGraph:
    event_id: str
    prefix_step: int
    constraint_ids: tuple[str, ...]
    option_ids: tuple[str, ...]
    edges: tuple[OptionEvidenceBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("graph event_id is required")
        if isinstance(self.prefix_step, bool) or not isinstance(self.prefix_step, int) or self.prefix_step < 0:
            raise ValueError("graph prefix_step is invalid")
        constraints = tuple(str(item) for item in self.constraint_ids)
        options = tuple(str(item) for item in self.option_ids)
        if not constraints or len(constraints) != len(set(constraints)) or any(not item for item in constraints):
            raise ValueError("graph constraint IDs must be unique and nonempty")
        if not options or len(options) != len(set(options)) or any(not item for item in options):
            raise ValueError("graph option IDs must be unique and nonempty")
        edges = tuple(self.edges)
        if any(not isinstance(edge, OptionEvidenceBinding) for edge in edges):
            raise TypeError("graph edges must be OptionEvidenceBinding values")
        object.__setattr__(self, "constraint_ids", constraints)
        object.__setattr__(self, "option_ids", options)
        object.__setattr__(self, "edges", edges)
        self.validate()

    def validate(self) -> None:
        constraint_set = set(self.constraint_ids)
        option_set = set(self.option_ids)
        seen: set[tuple[str, str, str, str]] = set()
        polarity: dict[tuple[str, str], set[BindingState]] = {}
        for edge in self.edges:
            if edge.event_id != self.event_id or edge.prefix_step != self.prefix_step:
                raise ValueError("edge is from a different event or prefix")
            if edge.constraint_id not in constraint_set or edge.option_id not in option_set:
                raise ValueError("edge references an unknown graph node")
            key = (edge.option_id, edge.constraint_id, edge.binding_state.value, edge.verification_source)
            if key in seen:
                raise ValueError("duplicate graph edge")
            seen.add(key)
            pair = (edge.option_id, edge.constraint_id)
            polarity.setdefault(pair, set()).add(edge.binding_state)
        for pair, states in polarity.items():
            if BindingState.SUPPORTS in states and BindingState.CONTRADICTS in states:
                raise ValueError(f"support/contradiction conflict for {pair}")

    def edges_for_constraint(self, constraint_id: str) -> tuple[OptionEvidenceBinding, ...]:
        return tuple(edge for edge in self.edges if edge.constraint_id == constraint_id)

    def edges_for_option(self, option_id: str) -> tuple[OptionEvidenceBinding, ...]:
        return tuple(edge for edge in self.edges if edge.option_id == option_id)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "revealnav-mf3zr-evidence-option-graph/1",
            "event_id": self.event_id,
            "prefix_step": self.prefix_step,
            "constraint_ids": list(self.constraint_ids),
            "option_ids": list(self.option_ids),
            "edges": [edge.as_mapping() for edge in self.edges],
        }


def graph_from_bindings(
    event_id: str,
    prefix_step: int,
    constraint_ids: Iterable[str],
    option_ids: Iterable[str],
    edges: Iterable[OptionEvidenceBinding],
) -> EvidenceOptionGraph:
    return EvidenceOptionGraph(
        event_id=event_id,
        prefix_step=prefix_step,
        constraint_ids=tuple(constraint_ids),
        option_ids=tuple(option_ids),
        edges=tuple(edges),
    )


__all__ = ["EvidenceOptionGraph", "graph_from_bindings"]
