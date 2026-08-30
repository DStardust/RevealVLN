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


def validate_action_identity(
    action_ids: Sequence[object],
    current_global_indices: Sequence[int],
    native_index: int,
    adapted_index: int | None = None,
    *,
    declared_native_id: str | None = None,
    declared_adapted_id: str | None = None,
    require_non_stop: bool = False,
) -> None:
    """Fail closed when local action identities do not round-trip globally.

    ETP uses index ``0`` for STOP and identifies navigable actions by a
    viewpoint ID.  A controller may rank IDs in a local candidate set, but
    execution must return to the same unique global index.  This helper keeps
    that invariant in one testable place.
    """

    ids = tuple(str(value) for value in action_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global action identity")
    current = tuple(int(value) for value in current_global_indices)
    if len(current) != len(set(current)) or any(
        value <= 0 or value >= len(ids) for value in current
    ):
        raise ValueError("invalid current global action indices")
    if not 0 <= int(native_index) < len(ids):
        raise ValueError("native action index is outside the global action set")
    native_index = int(native_index)
    if require_non_stop and native_index == 0:
        raise ValueError("an intervention cannot use STOP as the native action")
    if native_index != 0 and native_index not in current:
        raise ValueError("non-STOP native action is absent from current candidates")
    if declared_native_id is not None:
        if native_index == 0 or str(declared_native_id) != ids[native_index]:
            raise ValueError("declared native action ID does not round-trip")

    if adapted_index is None:
        if declared_adapted_id is not None:
            raise ValueError("adapted ID supplied without an adapted index")
        return
    if not 0 <= int(adapted_index) < len(ids):
        raise ValueError("adapted action index is outside the global action set")
    adapted_index = int(adapted_index)
    if adapted_index == 0 or adapted_index not in current:
        raise ValueError("adapted action is not a current navigable candidate")
    if declared_adapted_id is None or str(declared_adapted_id) != ids[adapted_index]:
        raise ValueError("declared adapted action ID does not round-trip")


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
