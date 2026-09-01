"""Strictly causal evidence memory with no outcome-bearing fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Iterable, Mapping


class EvidenceStatus(str, Enum):
    OBSERVED = "OBSERVED"
    SUPPORTED = "SUPPORTED"
    RESOLVED = "RESOLVED"
    CONTRADICTED = "CONTRADICTED"
    STALE = "STALE"


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex string")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    constraint_id: str
    step: int
    bbox_xyxy: tuple[float, float, float, float] | None
    candidate_id: str | None
    semantic_score: float
    first_seen_step: int
    last_seen_step: int
    status: EvidenceStatus
    observation_sha256: str
    annotation_sha256: str

    def __post_init__(self) -> None:
        for value, name in ((self.evidence_id, "evidence_id"), (self.constraint_id, "constraint_id")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for value, name in ((self.step, "step"), (self.first_seen_step, "first_seen_step"), (self.last_seen_step, "last_seen_step")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.first_seen_step > self.last_seen_step or self.step < self.first_seen_step or self.step > self.last_seen_step:
            raise ValueError("evidence step range is inconsistent")
        score = float(self.semantic_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("semantic_score must be finite in [0,1]")
        bbox = self.bbox_xyxy
        if bbox is not None:
            if len(bbox) != 4 or any(not math.isfinite(float(v)) for v in bbox):
                raise ValueError("bbox_xyxy must contain four finite values")
            if float(bbox[2]) < float(bbox[0]) or float(bbox[3]) < float(bbox[1]):
                raise ValueError("bbox coordinates must be ordered")
            bbox = tuple(float(v) for v in bbox)
        status = self.status if isinstance(self.status, EvidenceStatus) else EvidenceStatus(self.status)
        if self.candidate_id is not None and (not isinstance(self.candidate_id, str) or not self.candidate_id.strip()):
            raise ValueError("candidate_id must be non-empty when present")
        object.__setattr__(self, "semantic_score", score)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "status", status)
        _sha(self.observation_sha256, "observation_sha256")
        _sha(self.annotation_sha256, "annotation_sha256")

    def as_mapping(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "constraint_id": self.constraint_id,
            "step": self.step,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "candidate_id": self.candidate_id,
            "semantic_score": self.semantic_score,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "status": self.status.value,
            "observation_sha256": self.observation_sha256,
            "annotation_sha256": self.annotation_sha256,
        }


_FORBIDDEN = {
    "target", "delta_utility", "utility", "reward", "success", "spl", "ndtw",
    "sdtw", "catastrophe", "outcome", "navmesh", "pose", "future", "oracle",
}


def reject_forbidden_evidence_mapping(value: Mapping[str, object]) -> None:
    for key in value:
        lowered = str(key).casefold()
        if lowered in _FORBIDDEN or lowered.startswith(("future_", "outcome_", "oracle_", "treatment_")):
            raise ValueError(f"forbidden evidence field: {key}")


class EvidenceMemory:
    """Small mutable memory keyed by evidence identity."""

    def __init__(self, items: Iterable[EvidenceItem] = ()) -> None:
        self._items: dict[str, EvidenceItem] = {}
        for item in items:
            self.add(item)

    def add(self, item: EvidenceItem) -> None:
        if not isinstance(item, EvidenceItem):
            raise TypeError("EvidenceMemory accepts EvidenceItem values")
        prior = self._items.get(item.evidence_id)
        if prior is not None and prior != item:
            raise ValueError(f"evidence identity conflict: {item.evidence_id}")
        self._items[item.evidence_id] = item

    def update(self, items: Iterable[EvidenceItem], *, current_step: int | None = None) -> None:
        for item in items:
            self.add(item)
        if current_step is not None:
            self.mark_stale(int(current_step))

    def mark_stale(self, current_step: int) -> None:
        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        for key, item in tuple(self._items.items()):
            if item.last_seen_step < current_step and item.status not in (EvidenceStatus.CONTRADICTED, EvidenceStatus.STALE):
                self._items[key] = EvidenceItem(
                    **{**item.as_mapping(), "status": EvidenceStatus.STALE}
                )

    def items(self) -> tuple[EvidenceItem, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def for_constraint(self, constraint_id: str) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.items() if item.constraint_id == constraint_id)

    def for_option(self, constraint_ids: Iterable[str]) -> tuple[EvidenceItem, ...]:
        ids = set(str(value) for value in constraint_ids)
        return tuple(item for item in self.items() if item.constraint_id in ids)

    def resolved_constraints(self) -> tuple[str, ...]:
        return tuple(sorted({item.constraint_id for item in self.items() if item.status is EvidenceStatus.RESOLVED}))

    def snapshot(self) -> tuple[dict[str, object], ...]:
        return tuple(item.as_mapping() for item in self.items())


__all__ = ["EvidenceItem", "EvidenceMemory", "EvidenceStatus", "reject_forbidden_evidence_mapping"]
