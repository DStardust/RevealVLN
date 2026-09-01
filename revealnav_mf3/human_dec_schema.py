"""Blinded human DEC review schema for MF3ZP.

The schema deliberately contains no Qwen factor predictions or navigation
outcomes.  Completed labels are kept separate from the frozen Qwen records.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Sequence

from .evidence_uad import ConstraintState, derive_constraint_uad


class DecRole(str, Enum):
    DEC_REQUIRED = "DEC_REQUIRED"
    PREREQUISITE_ONLY = "PREREQUISITE_ONLY"
    FUTURE_NOT_RELEVANT = "FUTURE_NOT_RELEVANT"
    REDUNDANT = "REDUNDANT"
    INCORRECT = "INCORRECT"


class DecMatchType(str, Enum):
    EXACT_QWEN_ATOM = "EXACT_QWEN_ATOM"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    MISSING = "MISSING"


DEC_ROLES = {DecRole.DEC_REQUIRED, DecRole.PREREQUISITE_ONLY}
SCHEMA_VERSION = "revealnav-mf3zp-single-expert-dec-review/1"

_FORBIDDEN_KEYS = {
    "reward", "utility", "delta_utility", "outcome", "success", "spl",
    "ndtw", "sdtw", "catastrophe", "car_match", "correct_action",
    "native_action", "runner_action", "native_role", "runner_role",
    "qwen_preannotation", "normalized_constraints", "qwen_sge", "qwen_uad",
    "qwen_resolved", "qwen_rationale", "model_prediction", "fold_result",
}


class HumanDecSchemaError(ValueError):
    pass


def reject_blinded_payload(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in _FORBIDDEN_KEYS or lowered.startswith(
                ("reward_", "outcome_", "target_", "future_", "oracle_", "treatment_")
            ):
                raise HumanDecSchemaError(f"forbidden review field at {path}.{key}")
            reject_blinded_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_blinded_payload(child, path=f"{path}[{index}]")


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanDecSchemaError(f"{name} must be non-empty text")
    return value.strip()


def _validate_factor_rows(
    value: object,
    *,
    expected_steps: Sequence[int],
    require_complete: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(expected_steps):
        raise HumanDecSchemaError("factor_by_step does not match review window")
    normalized: list[dict[str, object]] = []
    for expected_step, row in zip(expected_steps, value, strict=True):
        required = {"step", "instantiated", "distinguishable", "resolved"}
        if not isinstance(row, Mapping) or set(row) != required or row["step"] != expected_step:
            raise HumanDecSchemaError("factor row schema/order mismatch")
        factors = [row[key] for key in ("instantiated", "distinguishable", "resolved")]
        if require_complete:
            if any(type(item) is not bool for item in factors):
                raise HumanDecSchemaError("completed DEC factor must be boolean")
        elif any(item is not None for item in factors):
            raise HumanDecSchemaError("blank review template contains a factor label")
        normalized.append(dict(row))
    return normalized


def _validate_missing_items(
    value: object,
    *,
    graph_ids: set[str],
    expected_steps: Sequence[int],
    require_complete: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise HumanDecSchemaError("missing_dec_constraints must be a list")
    if not require_complete and value:
        raise HumanDecSchemaError("blank template cannot contain a missing DEC label")
    seen: set[str] = set()
    normalized = []
    required = {
        "human_dec_item_id", "role", "text", "qwen_constraint_id",
        "match_type", "factor_by_step", "note",
    }
    for item in value:
        if not isinstance(item, Mapping) or set(item) != required:
            raise HumanDecSchemaError("missing DEC item schema mismatch")
        item_id = _nonempty_text(item["human_dec_item_id"], "human_dec_item_id")
        if item_id in seen:
            raise HumanDecSchemaError("duplicate human DEC item ID")
        seen.add(item_id)
        if item["role"] != "MISSING_DEC_CONSTRAINT":
            raise HumanDecSchemaError("missing DEC role mismatch")
        _nonempty_text(item["text"], "missing DEC text")
        qwen_id = item["qwen_constraint_id"]
        if qwen_id is not None and str(qwen_id) not in graph_ids:
            raise HumanDecSchemaError("missing DEC mapping references unknown Qwen constraint")
        match = DecMatchType(item["match_type"])
        if (qwen_id is None) != (match is DecMatchType.MISSING):
            raise HumanDecSchemaError("missing DEC mapping type disagrees with Qwen mapping")
        if not isinstance(item["note"], str):
            raise HumanDecSchemaError("missing DEC note must be text")
        _validate_factor_rows(
            item["factor_by_step"],
            expected_steps=expected_steps,
            require_complete=require_complete,
        )
        normalized.append(dict(item))
    return normalized


def validate_review_row(
    value: object,
    *,
    require_complete: bool,
    expected_mode: str | None = None,
) -> dict[str, object]:
    reject_blinded_payload(value)
    required = {
        "schema_version", "review_mode", "reviewer_id",
        "reviewer_blinded_to_outcomes", "reviewer_blinded_to_qwen_factors",
        "event_id", "dataset", "scene_id", "episode_id", "instruction",
        "decision_step", "current_candidate_ids", "review_prefix_start",
        "review_prefix_end", "extra_historical_evidence_steps", "prefixes",
        "constraint_graph_sha256", "constraint_graph", "constraint_reviews",
        "missing_dec_constraints", "dec_mapping", "review_complete",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise HumanDecSchemaError("single-expert review row schema mismatch")
    if value["schema_version"] != SCHEMA_VERSION:
        raise HumanDecSchemaError("review schema version mismatch")
    mode = str(value["review_mode"])
    if mode not in {"first", "retest"} or expected_mode is not None and mode != expected_mode:
        raise HumanDecSchemaError("review mode mismatch")
    if value["reviewer_blinded_to_outcomes"] is not True or value["reviewer_blinded_to_qwen_factors"] is not True:
        raise HumanDecSchemaError("review blinding declaration missing")
    for key in ("event_id", "dataset", "scene_id", "episode_id", "instruction", "constraint_graph_sha256"):
        _nonempty_text(value[key], key)
    if value["dataset"] not in {"R2R", "RxR"}:
        raise HumanDecSchemaError("unexpected review domain")
    decision = value["decision_step"]
    start = value["review_prefix_start"]
    end = value["review_prefix_end"]
    if any(type(item) is not int for item in (decision, start, end)) or end != decision or start < 0 or start > end:
        raise HumanDecSchemaError("invalid review prefix window")
    expected_steps = list(range(start, end + 1))
    candidates = value["current_candidate_ids"]
    if not isinstance(candidates, list) or len(candidates) != len(set(map(str, candidates))):
        raise HumanDecSchemaError("candidate aliases must be unique")
    extras = value["extra_historical_evidence_steps"]
    if not isinstance(extras, list) or len(extras) != len(set(extras)) or any(
        type(step) is not int or step < 0 or step >= start or step > decision for step in extras
    ):
        raise HumanDecSchemaError("extra historical evidence is not strictly causal")
    prefixes = value["prefixes"]
    if not isinstance(prefixes, list) or [row.get("step") for row in prefixes if isinstance(row, Mapping)] != expected_steps:
        raise HumanDecSchemaError("prefix references do not match review window")
    prefix_keys = {"step", "causal_storyboard_path", "current_panorama_path", "candidate_ids"}
    for prefix in prefixes:
        if not isinstance(prefix, Mapping) or set(prefix) != prefix_keys:
            raise HumanDecSchemaError("prefix reference schema mismatch")
        if any(not isinstance(prefix[key], str) or not prefix[key] for key in ("causal_storyboard_path", "current_panorama_path")):
            raise HumanDecSchemaError("prefix image reference missing")
    graph = value["constraint_graph"]
    if not isinstance(graph, list) or not graph:
        raise HumanDecSchemaError("constraint graph missing")
    graph_ids = [str(item.get("constraint_id")) for item in graph if isinstance(item, Mapping)]
    if len(graph_ids) != len(graph) or len(set(graph_ids)) != len(graph_ids):
        raise HumanDecSchemaError("constraint graph IDs invalid")
    reviews = value["constraint_reviews"]
    if not isinstance(reviews, Mapping) or set(reviews) != set(graph_ids):
        raise HumanDecSchemaError("constraint review population mismatch")
    human_dec_ids: set[str] = set()
    for cid in graph_ids:
        item = reviews[cid]
        item_keys = {"dec_role", "factor_by_step", "note"}
        if not isinstance(item, Mapping) or set(item) != item_keys or not isinstance(item["note"], str):
            raise HumanDecSchemaError("constraint review schema mismatch")
        if require_complete:
            role = DecRole(item["dec_role"])
            is_dec = role in DEC_ROLES
            _validate_factor_rows(
                item["factor_by_step"], expected_steps=expected_steps,
                require_complete=is_dec,
            )
            if not is_dec and any(
                row[key] is not None
                for row in item["factor_by_step"]
                for key in ("instantiated", "distinguishable", "resolved")
            ):
                raise HumanDecSchemaError("non-DEC constraint contains factor labels")
            if is_dec:
                human_dec_ids.add(f"human::{cid}")
        else:
            if item["dec_role"] is not None:
                raise HumanDecSchemaError("blank review template contains a DEC label")
            _validate_factor_rows(
                item["factor_by_step"], expected_steps=expected_steps,
                require_complete=False,
            )
    missing = _validate_missing_items(
        value["missing_dec_constraints"], graph_ids=set(graph_ids),
        expected_steps=expected_steps, require_complete=require_complete,
    )
    human_dec_ids.update(str(item["human_dec_item_id"]) for item in missing)
    mappings = value["dec_mapping"]
    if not isinstance(mappings, list):
        raise HumanDecSchemaError("DEC mapping must be a list")
    if not require_complete:
        if mappings:
            raise HumanDecSchemaError("blank template contains a DEC mapping")
    else:
        mapping_keys = {"human_dec_item_id", "qwen_constraint_id", "match_type"}
        seen_mapping: set[str] = set()
        for mapping in mappings:
            if not isinstance(mapping, Mapping) or set(mapping) != mapping_keys:
                raise HumanDecSchemaError("DEC mapping schema mismatch")
            item_id = _nonempty_text(mapping["human_dec_item_id"], "mapping human item")
            if item_id in seen_mapping:
                raise HumanDecSchemaError("duplicate DEC mapping")
            seen_mapping.add(item_id)
            qwen_id = mapping["qwen_constraint_id"]
            if qwen_id is not None and str(qwen_id) not in graph_ids:
                raise HumanDecSchemaError("DEC mapping references unknown Qwen constraint")
            match = DecMatchType(mapping["match_type"])
            if (qwen_id is None) != (match is DecMatchType.MISSING):
                raise HumanDecSchemaError("DEC mapping type mismatch")
        if seen_mapping != human_dec_ids:
            raise HumanDecSchemaError("DEC mapping is incomplete")
    reviewer_id = value["reviewer_id"]
    if require_complete:
        _nonempty_text(reviewer_id, "reviewer_id")
        if value["review_complete"] is not True:
            raise HumanDecSchemaError("completed review flag missing")
    elif reviewer_id != "" or value["review_complete"] is not False:
        raise HumanDecSchemaError("blank template contains reviewer completion state")
    return dict(value)


def terminal_uad(factor_by_step: Sequence[Mapping[str, object]]) -> ConstraintState:
    return derive_constraint_uad(
        (bool(row["instantiated"]) for row in factor_by_step),
        (bool(row["distinguishable"]) for row in factor_by_step),
        (bool(row["resolved"]) for row in factor_by_step),
        stability_k=3,
    )[-1]


__all__ = [
    "DEC_ROLES", "SCHEMA_VERSION", "DecMatchType", "DecRole",
    "HumanDecSchemaError", "reject_blinded_payload", "terminal_uad",
    "validate_review_row",
]
