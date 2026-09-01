"""Causal candidate-to-option identity helpers for MF3ZR.

Only opaque candidate IDs already present in the sealed causal prefixes are
used.  The helpers never infer a semantic/correct branch and never consult a
route, reward, or treatment result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

from .option_binding_schema import OptionIdentity, deterministic_option_id


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CandidatePersistence:
    candidate_id: str
    first_seen_step: int
    last_seen_step: int
    observed_steps: tuple[int, ...]
    persistence_status: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "observed_steps": list(self.observed_steps),
            "persistence_status": self.persistence_status,
        }


def anchor_checkpoint_id(event_id: str, step: int, candidate_id: str, prefix_commitment: str) -> str:
    """Create an anchor reference from the first causal observation only."""

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("anchor step must be non-negative")
    if not all(isinstance(value, str) and value for value in (event_id, candidate_id, prefix_commitment)):
        raise ValueError("anchor identity fields must be non-empty")
    return "anchor-" + _hash({
        "event_id": event_id,
        "step": step,
        "candidate_id": candidate_id,
        "prefix_commitment": prefix_commitment,
    })


def _normalise_prefixes(prefixes: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    if isinstance(prefixes, (str, bytes, bytearray)) or not prefixes:
        raise ValueError("causal prefix sequence is empty")
    rows: list[dict[str, object]] = []
    seen_steps: set[int] = set()
    for row in prefixes:
        if not isinstance(row, Mapping):
            raise TypeError("causal prefix must be a mapping")
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("causal prefix step is invalid")
        if step in seen_steps:
            raise ValueError("causal prefix repeats a step")
        candidates = row.get("candidate_ids")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("causal prefix has no candidate IDs")
        ids = tuple(str(value) for value in candidates)
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("causal candidate IDs must be unique and nonempty")
        # A source hash is a commitment to the image/storyboard and/or trace
        # reference.  We do not parse trace payloads here.
        ref = row.get("source_commitment")
        if not isinstance(ref, str) or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            ref = _hash({"step": step, "candidate_ids": list(ids)})
        rows.append({"step": int(step), "candidate_ids": ids, "source_commitment": ref})
        seen_steps.add(int(step))
    rows.sort(key=lambda item: int(item["step"]))
    # A review window may intentionally start after step zero.  It can still
    # describe the observed opaque aliases, but it cannot prove their true
    # birth/persistence before the window.  ``candidate_persistence`` marks
    # that limitation and the support gate rejects it; no hindsight fill-in
    # is performed.
    return tuple(rows)


def candidate_persistence(prefixes: Sequence[Mapping[str, object]]) -> tuple[CandidatePersistence, ...]:
    rows = _normalise_prefixes(prefixes)
    observed: dict[str, list[int]] = {}
    for row in rows:
        for candidate in row["candidate_ids"]:
            observed.setdefault(str(candidate), []).append(int(row["step"]))
    result = []
    for candidate_id in sorted(observed):
        steps = tuple(observed[candidate_id])
        contiguous = steps == tuple(range(steps[0], steps[-1] + 1))
        window_truncated = steps[0] != 0
        if window_truncated:
            status = "IDENTITY_WINDOW_TRUNCATED"
        else:
            status = "PERSISTENT_ALIAS" if contiguous else "IDENTITY_GAP_UNRESOLVED"
        result.append(CandidatePersistence(
            candidate_id=candidate_id,
            first_seen_step=steps[0],
            last_seen_step=steps[-1],
            observed_steps=steps,
            persistence_status=status,
        ))
    return tuple(result)


def build_option_identities(
    event_id: str,
    prefixes: Sequence[Mapping[str, object]],
) -> tuple[tuple[OptionIdentity, ...], tuple[str, ...]]:
    """Build deterministic option IDs and report unresolved persistence gaps."""

    rows = _normalise_prefixes(prefixes)
    persistence = candidate_persistence(rows)
    issues: list[str] = []
    identities: list[OptionIdentity] = []
    for item in persistence:
        option_id = deterministic_option_id(event_id, item.first_seen_step, item.candidate_id)
        prefix = next(row for row in rows if int(row["step"]) == item.first_seen_step)
        anchor = anchor_checkpoint_id(event_id, item.first_seen_step, item.candidate_id, str(prefix["source_commitment"]))
        status = "CAUSAL_OPAQUE_CANDIDATE"
        if item.persistence_status != "PERSISTENT_ALIAS":
            status = "OPTION_IDENTITY_UNRESOLVED"
            issues.append(f"{item.candidate_id}:candidate_persistence_gap")
        identities.append(OptionIdentity(
            option_id=option_id,
            event_id=event_id,
            candidate_id=item.candidate_id,
            first_seen_step=item.first_seen_step,
            last_seen_step=item.last_seen_step,
            anchor_checkpoint_id=anchor,
            identity_status=status,
        ))
    return (
        tuple(sorted(identities, key=lambda value: (value.first_seen_step, value.candidate_id))),
        tuple(sorted(set(issues))),
    )


def validate_binding_step(identity: OptionIdentity, step: int) -> None:
    """Reject a binding before causal candidate birth."""

    if isinstance(step, bool) or not isinstance(step, int) or step < identity.first_seen_step:
        raise ValueError("option binding occurs before candidate first_seen_step")
    if step > identity.last_seen_step:
        raise ValueError("option binding occurs after observed candidate persistence")


__all__ = [
    "CandidatePersistence", "anchor_checkpoint_id", "candidate_persistence",
    "build_option_identities", "validate_binding_step",
]
