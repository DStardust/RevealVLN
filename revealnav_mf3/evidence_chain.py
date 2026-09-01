"""Dependency-closed decisive evidence chains."""

from __future__ import annotations

from .evidence_constraints import InstructionEvidenceGraph


def decisive_chain(
    graph: InstructionEvidenceGraph, option_id: str,
) -> tuple[str, ...]:
    """Return the transitive dependency closure required by an option."""

    required = set(graph.required_for_option(option_id))
    by_id = {value.constraint_id: value for value in graph.constraints}
    pending = list(required)
    while pending:
        current = pending.pop()
        for dependency in by_id[current].dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return tuple(cid for cid in graph.topological_order() if cid in required)


def option_dependencies(
    graph: InstructionEvidenceGraph, option_id: str,
) -> tuple[str, ...]:
    """Alias with a name useful at policy call sites."""

    return decisive_chain(graph, option_id)


__all__ = ["decisive_chain", "option_dependencies"]
