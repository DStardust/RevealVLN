"""Instruction-level decisive evidence graphs for RevealSkill.

The graph is an annotation/provenance object, not a policy.  It contains no
reward, trajectory outcome, or simulator state and can therefore be safely
stored beside a causal observation record.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping


class ConstraintKind(str, Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    DIRECTION = "DIRECTION"
    ORDINAL = "ORDINAL"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    EXCLUSION = "EXCLUSION"
    GOAL = "GOAL"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _ids(values: Iterable[object], name: str) -> tuple[str, ...]:
    try:
        result = tuple(_text(value, name) for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers")
    return result


@dataclass(frozen=True)
class EvidenceConstraint:
    constraint_id: str
    kind: ConstraintKind
    subject: str
    relation: str | None
    object: str | None
    dependencies: tuple[str, ...]
    decisive_for: tuple[str, ...]

    def __post_init__(self) -> None:
        cid = _text(self.constraint_id, "constraint_id")
        kind = self.kind if isinstance(self.kind, ConstraintKind) else ConstraintKind(self.kind)
        subject = _text(self.subject, "subject")
        relation = None if self.relation is None else _text(self.relation, "relation")
        obj = None if self.object is None else _text(self.object, "object")
        dependencies = _ids(self.dependencies, "dependencies")
        decisive_for = _ids(self.decisive_for, "decisive_for")
        object.__setattr__(self, "constraint_id", cid)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "object", obj)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "decisive_for", decisive_for)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidenceConstraint":
        if not isinstance(value, Mapping):
            raise TypeError("constraint must be a mapping")
        allowed = {
            "constraint_id", "kind", "subject", "relation", "object",
            "dependencies", "decisive_for",
        }
        if set(value) != allowed:
            raise ValueError(
                f"constraint schema mismatch; missing={sorted(allowed-set(value))}, "
                f"unexpected={sorted(set(value)-allowed)}"
            )
        return cls(
            constraint_id=value["constraint_id"],
            kind=value["kind"],
            subject=value["subject"],
            relation=value["relation"],
            object=value["object"],
            dependencies=tuple(value["dependencies"]),
            decisive_for=tuple(value["decisive_for"]),
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "dependencies": list(self.dependencies),
            "decisive_for": list(self.decisive_for),
        }


@dataclass(frozen=True)
class InstructionEvidenceGraph:
    instruction: str
    constraints: tuple[EvidenceConstraint, ...]
    parser_model: str
    parser_prompt_sha256: str

    def __post_init__(self) -> None:
        instruction = _text(self.instruction, "instruction")
        constraints = tuple(self.constraints)
        if not constraints:
            raise ValueError("evidence graph must contain at least one constraint")
        if any(not isinstance(value, EvidenceConstraint) for value in constraints):
            raise TypeError("constraints must contain EvidenceConstraint values")
        ids = [value.constraint_id for value in constraints]
        if len(set(ids)) != len(ids):
            raise ValueError("constraint IDs must be unique")
        _text(self.parser_model, "parser_model")
        prompt_hash = _text(self.parser_prompt_sha256, "parser_prompt_sha256")
        if len(prompt_hash) != 64 or any(char not in "0123456789abcdef" for char in prompt_hash):
            raise ValueError("parser_prompt_sha256 must be a lowercase SHA-256")
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "constraints", constraints)
        self.validate_dag()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        instruction: str,
        parser_model: str,
        parser_prompt_sha256: str,
    ) -> "InstructionEvidenceGraph":
        if not isinstance(value, Mapping) or set(value) not in ({"constraints"}, {"constraints", "dependencies"}):
            raise ValueError("graph annotation must contain constraints and optional dependency edges")
        raw = value["constraints"]
        if not isinstance(raw, list):
            raise TypeError("graph constraints must be a list")
        graph = cls(
            instruction=instruction,
            constraints=tuple(EvidenceConstraint.from_mapping(item) for item in raw),
            parser_model=parser_model,
            parser_prompt_sha256=parser_prompt_sha256,
        )
        if "dependencies" in value:
            edges = value["dependencies"]
            if not isinstance(edges, list):
                raise TypeError("graph dependencies must be a list")
            observed: set[tuple[str, str]] = set()
            for edge in edges:
                if not isinstance(edge, Mapping) or set(edge) != {"from", "to"}:
                    raise ValueError("dependency edge must contain from/to")
                observed.add((str(edge["from"]), str(edge["to"])))
            expected = {
                (dependency, constraint.constraint_id)
                for constraint in graph.constraints
                for dependency in constraint.dependencies
            }
            if observed != expected or len(observed) != len(edges):
                raise ValueError("top-level dependency edges disagree with constraints")
        return graph

    def validate_dag(self) -> None:
        ids = {value.constraint_id for value in self.constraints}
        indegree = {value.constraint_id: 0 for value in self.constraints}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for constraint in self.constraints:
            for dependency in constraint.dependencies:
                if dependency not in ids:
                    raise ValueError(
                        f"constraint {constraint.constraint_id} depends on unknown {dependency}"
                    )
                outgoing[dependency].append(constraint.constraint_id)
                indegree[constraint.constraint_id] += 1
        queue = deque(sorted(cid for cid, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for child in sorted(outgoing[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(ids):
            raise ValueError("instruction evidence graph contains a cycle")

    def _by_id(self) -> dict[str, EvidenceConstraint]:
        return {value.constraint_id: value for value in self.constraints}

    def topological_order(self) -> tuple[str, ...]:
        self.validate_dag()
        by_id = self._by_id()
        pending = {cid: set(value.dependencies) for cid, value in by_id.items()}
        result: list[str] = []
        while pending:
            ready = sorted(cid for cid, deps in pending.items() if not deps)
            if not ready:
                raise ValueError("graph is cyclic")
            result.extend(ready)
            for cid in ready:
                pending.pop(cid)
            for deps in pending.values():
                deps.difference_update(ready)
        return tuple(result)

    def resolved_constraints(self, states: Mapping[str, object]) -> tuple[str, ...]:
        by_id = self._by_id()
        if any(str(key) not in by_id for key in states):
            raise ValueError("state contains an unknown constraint")
        return tuple(
            cid for cid in self.topological_order()
            if str(getattr(states.get(cid), "value", states.get(cid))) == "D"
        )

    def active_frontier(self, states: Mapping[str, object]) -> tuple[str, ...]:
        resolved = set(self.resolved_constraints(states))
        by_id = self._by_id()
        return tuple(
            cid for cid in self.topological_order()
            if cid not in resolved and set(by_id[cid].dependencies).issubset(resolved)
        )

    def required_for_option(self, option_id: str) -> tuple[str, ...]:
        option = _text(option_id, "option_id")
        return tuple(
            cid for cid in self.topological_order()
            if option in self._by_id()[cid].decisive_for
        )

    def canonical_sha256(self) -> str:
        payload = {
            "instruction": self.instruction,
            "parser_model": self.parser_model,
            "parser_prompt_sha256": self.parser_prompt_sha256,
            "constraints": [value.as_mapping() for value in self.constraints],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()


__all__ = ["ConstraintKind", "EvidenceConstraint", "InstructionEvidenceGraph"]
