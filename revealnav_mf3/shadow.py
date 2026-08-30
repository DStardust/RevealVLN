"""Pure bookkeeping for the MF3B action-preserving shadow gate."""

from __future__ import annotations

from collections.abc import Collection, Sequence


SHADOW_OUTCOMES = {
    "RESCUE",
    "HARM",
    "AGREE_CORRECT",
    "AGREE_INCORRECT",
    "DISAGREE_NEITHER",
    "INELIGIBLE",
}


def current_local_action_indices(
    action_ids: Sequence[object],
    action_mask: Sequence[bool],
    visited_mask: Sequence[bool],
    local_action_ids: Collection[str],
) -> tuple[int, ...]:
    """Return current-local, unvisited actions from ETP's native graph."""

    if not (
        len(action_ids) == len(action_mask) == len(visited_mask)
    ):
        raise ValueError("invalid ETP candidate identity inputs")
    selected = []
    seen = set()
    for index in range(1, len(action_ids)):
        action_id = action_ids[index]
        if action_id is None or not action_mask[index] or visited_mask[index]:
            continue
        identity = str(action_id)
        if identity in seen:
            raise ValueError("duplicate ETP action identity")
        seen.add(identity)
        if identity in local_action_ids:
            selected.append(index)
    return tuple(selected)


def classify_shadow_outcome(
    native_action: int,
    uad_action: int,
    teacher_action: int,
    current_local_indices: Sequence[int],
) -> str:
    """Classify one label-only UAD/native comparison."""

    current = set(current_local_indices)
    if teacher_action not in current or uad_action not in current:
        return "INELIGIBLE"
    native_correct = native_action == teacher_action
    uad_correct = uad_action == teacher_action
    if not native_correct and uad_correct:
        return "RESCUE"
    if native_correct and not uad_correct:
        return "HARM"
    if native_correct:
        return "AGREE_CORRECT"
    if native_action == uad_action:
        return "AGREE_INCORRECT"
    return "DISAGREE_NEITHER"
