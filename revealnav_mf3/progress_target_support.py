"""Exact, outcome-blind local decision target validation for MF3ZV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .progress_schema import reject_forbidden_progress_payload


@dataclass(frozen=True)
class ExactLocalTarget:
    dataset: str
    episode_id: str
    scene_id: str
    decision_step: int
    candidate_action_ids: tuple[str, ...]
    native_action_id: str
    source_trace_sha256: str
    support_rule: str = "exact_same_episode_prefix_native_action_in_dynamic_candidate_set"

    def __post_init__(self) -> None:
        if self.decision_step < 0:
            raise ValueError("decision_step must be non-negative")
        if not self.candidate_action_ids:
            raise ValueError("candidate set must be non-empty")
        if len(set(self.candidate_action_ids)) != len(self.candidate_action_ids):
            raise ValueError("candidate IDs must be unique")
        if self.native_action_id not in self.candidate_action_ids:
            raise ValueError("native action must be in the exact dynamic candidate set")
        if len(self.source_trace_sha256) != 64:
            raise ValueError("source trace SHA-256 is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_action_ids"] = list(self.candidate_action_ids)
        return data


SAFE_TRACE_KEYS = frozenset(
    {
        "step",
        "current_local_action_ids",
        "native_action_id",
        "native_action_index",
        "feature_native_action_id",
        "schema_version",
        "record_hash",
        "previous_hash",
        "public_unseen_authorized",
    }
)


def exact_target_from_trace_row(
    *, dataset: str, episode_id: str, scene_id: str, row: Mapping[str, Any], source_sha256: str
) -> ExactLocalTarget:
    safe = {key: row[key] for key in SAFE_TRACE_KEYS if key in row}
    reject_forbidden_progress_payload(safe)
    if row.get("public_unseen_authorized") not in {None, False}:
        raise ValueError("public split authorization must remain false")
    return ExactLocalTarget(
        dataset=dataset,
        episode_id=episode_id,
        scene_id=scene_id,
        decision_step=int(safe["step"]),
        candidate_action_ids=tuple(str(item) for item in safe["current_local_action_ids"]),
        native_action_id=str(safe["native_action_id"]),
        source_trace_sha256=source_sha256,
    )

