"""High-recall language screening for public VLN-CE Reveal candidates."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Iterator, Mapping


_PATTERNS = {
    "ordinal": re.compile(r"\b(first|second|third|next|another|last)\b", re.I),
    "exclusion": re.compile(
        r"\b(skip|skipping|ignore|pass|past|not the first|do not take)\b", re.I
    ),
    "temporal": re.compile(r"\b(after|before|once|until|then|when)\b", re.I),
    "branch": re.compile(
        r"\b(left|right|turn|door|doorway|entrance|hallway|corridor|stair|stairs)\b",
        re.I,
    ),
}


@dataclass(frozen=True)
class ScreenedInstruction:
    dataset: str
    split: str
    episode_id: str
    instruction_id: str
    trajectory_id: str
    scene_id: str
    language: str
    instruction: str
    triggers: tuple[str, ...]


def trigger_types(instruction: str) -> tuple[str, ...]:
    """Return auditable lexical triggers; this is not a final event label."""

    return tuple(
        name for name, pattern in _PATTERNS.items() if pattern.search(instruction)
    )


def is_reveal_candidate(triggers: Iterable[str]) -> bool:
    trigger_set = set(triggers)
    return "branch" in trigger_set and bool(
        trigger_set.intersection({"ordinal", "exclusion", "temporal"})
    )


def iter_vlnce_episodes(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Read the official gzipped Habitat dataset format."""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError("VLN-CE annotation must contain an episodes list")
    for index, episode in enumerate(payload["episodes"]):
        if not isinstance(episode, dict):
            raise ValueError(f"episode {index} must be a JSON object")
        yield episode


def screen_vlnce(
    episodes: Iterable[Mapping[str, object]],
    *,
    dataset: str,
    split: str,
    languages: set[str] | None = None,
) -> Iterator[ScreenedInstruction]:
    """Screen official RxR-CE/R2R-CE metadata without future observations."""

    if dataset not in {"rxr-ce", "r2r-ce"}:
        raise ValueError("dataset must be rxr-ce or r2r-ce")
    if split not in {"train", "val_seen"}:
        raise ValueError("Phase 0 screening is restricted to train and val_seen")
    allowed_languages = {"en-US", "en-IN"} if dataset == "rxr-ce" else {"en"}
    if languages is not None and not languages.issubset(allowed_languages):
        raise ValueError(
            f"languages for {dataset} must be a subset of {sorted(allowed_languages)}"
        )
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("episode must be a mapping")
        instruction = episode.get("instruction")
        if not isinstance(instruction, dict):
            raise ValueError("episode instruction must be a JSON object")
        episode_id_raw = episode.get("episode_id")
        trajectory_id_raw = episode.get("trajectory_id")
        scene_id_raw = episode.get("scene_id")
        if dataset == "rxr-ce":
            if not isinstance(episode_id_raw, str) or not episode_id_raw:
                raise ValueError("RxR-CE episode_id must be a non-empty string")
            if not isinstance(trajectory_id_raw, str) or not trajectory_id_raw:
                raise ValueError("RxR-CE trajectory_id must be a non-empty string")
            instruction_id_raw = instruction.get("instruction_id")
            language_raw = instruction.get("language")
            if not isinstance(instruction_id_raw, str) or not instruction_id_raw:
                raise ValueError(
                    "RxR-CE instruction_id must be a non-empty string"
                )
            if not isinstance(language_raw, str) or not language_raw:
                raise ValueError("RxR-CE language must be a non-empty string")
            if language_raw not in allowed_languages:
                continue
            instruction_id = instruction_id_raw
            language = language_raw
        else:
            if not isinstance(episode_id_raw, int) or isinstance(
                episode_id_raw, bool
            ):
                raise ValueError("R2R-CE episode_id must be an integer")
            if not isinstance(trajectory_id_raw, int) or isinstance(
                trajectory_id_raw, bool
            ):
                raise ValueError("R2R-CE trajectory_id must be an integer")
            if "language" in instruction or "instruction_id" in instruction:
                raise ValueError(
                    "R2R-CE instruction must use the official implicit-English schema"
                )
            instruction_tokens = instruction.get("instruction_tokens")
            if not isinstance(instruction_tokens, list) or any(
                not isinstance(token, int) or isinstance(token, bool)
                for token in instruction_tokens
            ):
                raise ValueError(
                    "R2R-CE instruction_tokens must be an integer list"
                )
            instruction_id = f"{episode_id_raw}:0"
            language = "en"
        if languages is not None and language not in languages:
            continue
        text = instruction.get("instruction_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("instruction_text must be a non-empty string")
        if not isinstance(scene_id_raw, str) or not scene_id_raw:
            raise ValueError("scene_id must be a non-empty string")
        scene_path = Path(scene_id_raw)
        if scene_path.suffix != ".glb" or not scene_path.stem:
            raise ValueError("scene_id must identify an MP3D .glb scene")
        triggers = trigger_types(text)
        if not is_reveal_candidate(triggers):
            continue
        yield ScreenedInstruction(
            dataset=dataset,
            split=split,
            episode_id=str(episode_id_raw),
            instruction_id=instruction_id,
            trajectory_id=str(trajectory_id_raw),
            scene_id=scene_path.stem,
            language=language,
            instruction=text,
            triggers=triggers,
        )


def screening_summary(
    candidates: Iterable[ScreenedInstruction],
) -> dict[str, object]:
    items = list(candidates)
    trigger_counts = Counter(
        trigger for candidate in items for trigger in candidate.triggers
    )
    return {
        "candidate_instructions": len(items),
        "unique_trajectories": len(
            {(item.scene_id, item.trajectory_id) for item in items}
        ),
        "unique_scenes": len({item.scene_id for item in items}),
        "languages": dict(sorted(Counter(item.language for item in items).items())),
        "triggers": dict(sorted(trigger_counts.items())),
    }


def pilot_sample(
    candidates: Iterable[ScreenedInstruction], count: int, seed: int = 0
) -> tuple[ScreenedInstruction, ...]:
    """Uniformly sample unique trajectories, then one instruction per trajectory."""

    if count < 0:
        raise ValueError("count must be non-negative")
    random = Random(seed)
    groups: dict[tuple[str, str], list[ScreenedInstruction]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.scene_id, candidate.trajectory_id), []).append(
            candidate
        )
    for trajectory_key in sorted(groups):
        group = groups[trajectory_key]
        group.sort(
            key=lambda candidate: (
                candidate.dataset,
                candidate.split,
                candidate.instruction_id,
                candidate.episode_id,
                candidate.triggers,
                candidate.instruction,
            )
        )
    trajectory_keys = sorted(groups)
    if len(trajectory_keys) < count:
        raise ValueError(
            f"requested {count} unique trajectories, only {len(trajectory_keys)} available"
        )
    selected_keys = random.sample(trajectory_keys, count)
    return tuple(random.choice(groups[key]) for key in selected_keys)
