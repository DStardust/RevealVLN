"""Correct S/G/E-independent validator for MF3ZP RevealSkill v1.1."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def validate_evidence_response_v1_1(
    value: object,
    *,
    active_constraint_ids: Sequence[str],
    allowed_candidate_ids: Sequence[str],
    image_count: int,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {"constraints"} or not isinstance(value["constraints"], Mapping):
        raise ValueError("evidence response must contain only a constraints object")
    expected = {str(item) for item in active_constraint_ids}
    if set(value["constraints"]) != expected:
        raise ValueError("evidence response constraint set mismatch")
    allowed = {str(item) for item in allowed_candidate_ids}
    required = {"instantiated", "distinguishable", "resolved", "bbox_xyxy", "candidate_ids", "evidence_image_indices", "evidence"}
    normalized = {}
    for cid, raw in value["constraints"].items():
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"constraint response schema mismatch: {cid}")
        factors = tuple(raw[key] for key in ("instantiated", "distinguishable", "resolved"))
        if any(type(value) is not bool for value in factors):
            raise ValueError(f"constraint factor types invalid: {cid}")
        bbox = raw["bbox_xyxy"]
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in bbox):
                raise ValueError(f"invalid bbox: {cid}")
            if float(bbox[2]) < float(bbox[0]) or float(bbox[3]) < float(bbox[1]):
                raise ValueError(f"unordered bbox: {cid}")
        candidate_ids = raw["candidate_ids"]
        if not isinstance(candidate_ids, list) or len(candidate_ids) != len(set(str(item) for item in candidate_ids)) or any(str(item) not in allowed for item in candidate_ids):
            raise ValueError(f"unknown/duplicate candidate binding: {cid}")
        indices = raw["evidence_image_indices"]
        if not isinstance(indices, list) or len(indices) != len(set(indices)) or any(type(item) is not int or item < 0 or item >= image_count for item in indices):
            raise ValueError(f"invalid evidence image index: {cid}")
        evidence = raw["evidence"]
        if not isinstance(evidence, str) or len(evidence) > 2000 or (not evidence.strip() and any(factors)):
            raise ValueError(f"invalid evidence explanation: {cid}")
        normalized[str(cid)] = {
            "instantiated": factors[0], "distinguishable": factors[1], "resolved": factors[2],
            "bbox_xyxy": bbox, "candidate_ids": [str(item) for item in candidate_ids],
            "evidence_image_indices": list(indices), "evidence": evidence.strip(),
        }
    return normalized


__all__ = ["validate_evidence_response_v1_1"]
