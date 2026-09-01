"""Explicit evidence-to-option relations; no learned graph network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .evidence_constraints import InstructionEvidenceGraph
from .evidence_memory import EvidenceMemory
from .option_graph import OptionGraph


@dataclass(frozen=True)
class EvidenceOptionBinding:
    evidence_id: str
    option_id: str
    constraint_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.evidence_id, self.option_id, self.constraint_id)):
            raise ValueError("binding identifiers must be non-empty")


def validate_bindings(
    bindings: Iterable[EvidenceOptionBinding],
    evidence: EvidenceMemory,
    options: OptionGraph,
    graph: InstructionEvidenceGraph,
) -> tuple[EvidenceOptionBinding, ...]:
    result = tuple(bindings)
    seen: set[tuple[str, str, str]] = set()
    for binding in result:
        key = (binding.evidence_id, binding.option_id, binding.constraint_id)
        if key in seen:
            raise ValueError("duplicate evidence-option binding")
        seen.add(key)
        if not evidence.for_constraint(binding.constraint_id) or not any(item.evidence_id == binding.evidence_id for item in evidence.items()):
            raise ValueError("binding references unknown evidence")
        node = options.get(binding.option_id)
        if binding.constraint_id not in graph.required_for_option(node.branch_candidate_id) and binding.constraint_id not in node.required_constraints:
            raise ValueError("binding constraint is not required for option")
    return result


__all__ = ["EvidenceOptionBinding", "validate_bindings"]
