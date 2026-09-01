"""Outcome-blind Reveal Event records mined from causal observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


EVENT_SCHEMA = "revealnav-mf3zp-reveal-event/1"


@dataclass(frozen=True)
class RevealEvent:
    dataset: str
    scene_id: str
    episode_id: str
    event_id: str
    instruction: str
    constraint_graph_sha256: str | None
    prefix_start: int
    prefix_end: int
    causal_prefix_sha256: str
    option_ids: tuple[str, ...]
    source_request_id: str
    observation_path: str
    current_panorama_path: str
    trigger_types: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, name in ((self.dataset, "dataset"), (self.scene_id, "scene_id"), (self.episode_id, "episode_id"), (self.event_id, "event_id"), (self.instruction, "instruction"), (self.causal_prefix_sha256, "causal_prefix_sha256"), (self.source_request_id, "source_request_id"), (self.observation_path, "observation_path"), (self.current_panorama_path, "current_panorama_path")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.dataset not in {"R2R", "RxR"}:
            raise ValueError("unknown dataset")
        for value, name in ((self.prefix_start, "prefix_start"), (self.prefix_end, "prefix_end")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative integer")
        if self.prefix_start > self.prefix_end:
            raise ValueError("prefix interval is invalid")
        for value, name in ((self.causal_prefix_sha256, "causal_prefix_sha256"),):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if self.constraint_graph_sha256 is not None and (len(self.constraint_graph_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.constraint_graph_sha256)):
            raise ValueError("constraint_graph_sha256 must be lowercase SHA-256 when present")
        options = tuple(str(value) for value in self.option_ids)
        triggers = tuple(str(value) for value in self.trigger_types)
        if len(options) != len(set(options)) or any(not value for value in options):
            raise ValueError("option_ids must be unique and nonempty")
        if not triggers or len(triggers) != len(set(triggers)):
            raise ValueError("trigger_types must be nonempty and unique")
        object.__setattr__(self, "option_ids", options)
        object.__setattr__(self, "trigger_types", triggers)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": EVENT_SCHEMA,
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "event_id": self.event_id,
            "instruction": self.instruction,
            "constraint_graph_sha256": self.constraint_graph_sha256,
            "prefix_start": self.prefix_start,
            "prefix_end": self.prefix_end,
            "causal_prefix_sha256": self.causal_prefix_sha256,
            "option_ids": list(self.option_ids),
            "source_request_id": self.source_request_id,
            "observation_path": self.observation_path,
            "current_panorama_path": self.current_panorama_path,
            "trigger_types": list(self.trigger_types),
        }


def canonical_event_id(dataset: str, scene_id: str, episode_id: str, prefix_end: int, trigger_types: Iterable[str]) -> str:
    payload = {
        "dataset": str(dataset), "scene_id": str(scene_id), "episode_id": str(episode_id),
        "prefix_end": int(prefix_end), "trigger_types": sorted(str(value) for value in trigger_types),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["EVENT_SCHEMA", "RevealEvent", "canonical_event_id"]
