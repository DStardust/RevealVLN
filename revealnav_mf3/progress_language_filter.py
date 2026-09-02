"""Fixed lexical discovery and scene-balanced selection for MF3ZV."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from .progress_schema import AtomReviewStatus, ProgressAtom, ProgressFamily


SELECTION_SALT = "mf3zv-minimal-progress-support-v1/selection/1"

_ORDINAL_VALUE = {
    "first": "1",
    "second": "2",
    "third": "3",
    "next": "NEXT",
    "another": "ANOTHER",
}

_NAV_OBJECT = (
    r"left(?:\s+turn)?|right(?:\s+turn)?|turn|door(?:way)?|entrance|opening|"
    r"hall(?:way)?|corridor|room|bedroom|bathroom|stair(?:s|case)?|flight|intersection"
)
_ORDINAL_RE = re.compile(
    rf"\b(?P<ordinal>first|second|third|next|another)\b"
    rf"(?P<middle>(?:\s+[a-z][a-z'-]*){{0,3}}?)\s+"
    rf"(?P<object>{_NAV_OBJECT})\b",
    re.IGNORECASE,
)
_MOTION_PAST_RE = re.compile(
    r"\b(?P<prefix>after\s+passing|once\s+you\s+pass|"
    r"(?:go|walk|move|continue|head|proceed|travel)(?:\s+\w+){0,2}\s+past|go\s+beyond)\s+"
    r"(?P<object>(?:the\s+)?[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,4})",
    re.IGNORECASE,
)
_AFTER_LANDMARK_RE = re.compile(
    r"\bafter\s+(?P<object>(?:the\s+)?[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3})",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:then|and|"
    r"before|until|where|which|that)\b|[,.;]",
    re.IGNORECASE,
)
_GENERIC_AFTER = frozenset(
    {
        "that",
        "this",
        "entering",
        "exiting",
        "turning",
        "walking",
        "going",
        "you",
        "which",
    }
)


@dataclass(frozen=True)
class InstructionRecord:
    dataset: str
    episode_id: str
    scene_id: str
    instruction: str
    language: Optional[str]


@dataclass(frozen=True)
class AtomProposal:
    dataset: str
    episode_id: str
    scene_id: str
    instruction: str
    language: Optional[str]
    atom: ProgressAtom
    span_start: int
    span_end: int
    mechanical_review_status: str
    mechanical_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction": self.instruction,
            "language": self.language,
            "atom": self.atom.to_dict(),
            "span_start": self.span_start,
            "span_end": self.span_end,
            "mechanical_review_status": self.mechanical_review_status,
            "mechanical_reason": self.mechanical_reason,
        }


def raw_scene_id(scene_id: str) -> str:
    name = str(scene_id).replace("\\", "/").rsplit("/", 1)[-1]
    return name.removesuffix(".glb")


def load_train_instructions(path: Path, dataset: str) -> Iterator[InstructionRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) != {"episodes"}:
        raise ValueError(f"unexpected train instruction payload keys: {sorted(payload)}")
    for row in payload["episodes"]:
        instruction = row["instruction"]
        language = instruction.get("language")
        if dataset == "RxR" and language not in {"en-IN", "en-US"}:
            continue
        yield InstructionRecord(
            dataset=dataset,
            episode_id=str(row["episode_id"]),
            scene_id=raw_scene_id(row["scene_id"]),
            instruction=str(instruction["instruction_text"]),
            language=None if language is None else str(language),
        )


def _trim_object(text: str) -> str:
    stop = _BOUNDARY_RE.search(text)
    if stop:
        text = text[: stop.start()]
    return re.sub(r"\s+", " ", text).strip(" -")


def _atom_id(record: InstructionRecord, family: str, start: int, span: str) -> str:
    payload = "\x1f".join(
        [record.dataset, record.episode_id, family, str(start), span.casefold()]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def propose_progress_atoms(record: InstructionRecord) -> list[AtomProposal]:
    proposals: list[AtomProposal] = []
    text = record.instruction
    for match in _ORDINAL_RE.finditer(text):
        ordinal = match.group("ordinal").casefold()
        middle = re.sub(r"\s+", " ", match.group("middle").casefold()).strip()
        if ordinal == "next" and middle.startswith("to"):
            continue
        obj = re.sub(r"\s+", " ", match.group("object").casefold())
        span = match.group(0).strip()
        navigation_context = bool(
            re.search(
                r"\b(?:take|enter|go|walk|turn|make|through|at|into|reach|exit)\b",
                text[max(0, match.start() - 35) : match.end() + 15],
                re.IGNORECASE,
            )
        )
        status = (
            AtomReviewStatus.VALID_PROGRESS_ATOM
            if navigation_context
            else AtomReviewStatus.AMBIGUOUS_PROGRESS_ATOM
        )
        reason = (
            "ordinal bound to a navigational object and local motion cue"
            if status is AtomReviewStatus.VALID_PROGRESS_ATOM
            else "ordinal may be a static descriptor or lacks a local sequential-motion cue"
        )
        atom = ProgressAtom(
            atom_id=_atom_id(record, ProgressFamily.ORDINAL.value, match.start(), span),
            family=ProgressFamily.ORDINAL.value,
            subject=obj.replace(" ", "_"),
            relation="COUNT_TARGET",
            target_value=_ORDINAL_VALUE[ordinal],
            instruction_span=span,
        )
        proposals.append(
            AtomProposal(
                record.dataset,
                record.episode_id,
                record.scene_id,
                text,
                record.language,
                atom,
                match.start(),
                match.end(),
                status.value,
                reason,
            )
        )

    occupied = [(item.span_start, item.span_end) for item in proposals]
    for regex, explicit in ((_MOTION_PAST_RE, True), (_AFTER_LANDMARK_RE, False)):
        for match in regex.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            obj = _trim_object(match.group("object")).casefold()
            if not obj:
                continue
            head = obj.removeprefix("the ").split()[0]
            ambiguous = head in _GENERIC_AFTER or (not explicit and len(obj.split()) > 3)
            status = (
                AtomReviewStatus.AMBIGUOUS_PROGRESS_ATOM
                if ambiguous
                else AtomReviewStatus.VALID_PROGRESS_ATOM
            )
            span = text[match.start() : match.start("object") + len(obj)].strip()
            atom = ProgressAtom(
                atom_id=_atom_id(
                    record, ProgressFamily.PASSED_LANDMARK.value, match.start(), span
                ),
                family=ProgressFamily.PASSED_LANDMARK.value,
                subject=obj.removeprefix("the ").replace(" ", "_"),
                relation="PASSED",
                target_value="true",
                instruction_span=span,
            )
            proposals.append(
                AtomProposal(
                    record.dataset,
                    record.episode_id,
                    record.scene_id,
                    text,
                    record.language,
                    atom,
                    match.start(),
                    match.start("object") + len(obj),
                    status.value,
                    (
                        "explicit causal motion-past landmark expression"
                        if explicit and not ambiguous
                        else "after/past expression with a concrete landmark"
                        if not ambiguous
                        else "after expression does not identify a stable visual landmark"
                    ),
                )
            )
            occupied.append((match.start(), match.end()))
    return sorted(proposals, key=lambda item: (item.span_start, item.atom.atom_id))


def earliest_atom(record: InstructionRecord) -> Optional[AtomProposal]:
    proposals = propose_progress_atoms(record)
    if not proposals:
        return None
    return proposals[0]


def deterministic_scene_round_robin(
    proposals: Sequence[AtomProposal], limit: int
) -> list[AtomProposal]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    by_scene: dict[str, list[AtomProposal]] = defaultdict(list)
    for proposal in proposals:
        by_scene[proposal.scene_id].append(proposal)
    for scene, rows in by_scene.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{SELECTION_SALT}:{row.dataset}:{row.episode_id}".encode("utf-8")
            ).hexdigest()
        )
    scenes = sorted(
        by_scene,
        key=lambda scene: hashlib.sha256(
            f"{SELECTION_SALT}:scene:{scene}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[AtomProposal] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for scene in scenes:
            rows = by_scene[scene]
            if round_index < len(rows):
                selected.append(rows[round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return selected
