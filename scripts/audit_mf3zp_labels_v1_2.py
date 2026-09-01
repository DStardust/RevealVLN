#!/usr/bin/env python3
"""Future MF3ZP formal audit: exactly three reviewers plus adjudication.

This command is intentionally not run by the single-expert scout.  It makes
the sealed 3-reviewer/1-adjudicator contract executable and fail-closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.evidence_uad import derive_constraint_uad  # noqa: E402
from revealnav_mf3.human_dec_schema import DecRole  # noqa: E402
from revealnav_mf3.single_expert_dec_scout import (  # noqa: E402
    PILOT_EVENTS,
    PUBLIC_CLOSED,
    ScoutError,
    atomic_json,
    atomic_jsonl,
    inventory,
    load_graph,
    read_json,
    read_jsonl,
    sha256_file,
    verify_scout_protocol,
)


SCHEMA = "revealnav-mf3zp-formal-dec-review/1.2"
ADJUDICATION_SCHEMA = "revealnav-mf3zp-formal-dec-adjudication/1.2"
RESULT_SCHEMA = "revealnav-mf3zp-label-validity-result/1.2"
GOLD_SCHEMA = "revealnav-mf3zp-adjudicated-gold/1.2"


class FormalAuditError(ScoutError):
    pass


def fleiss_kappa(items: Sequence[Sequence[str]], categories: Sequence[str]) -> float:
    if not items or any(len(ratings) != 3 for ratings in items):
        raise FormalAuditError("Fleiss kappa requires exactly three ratings per item")
    n = 3
    counts = [[ratings.count(category) for category in categories] for ratings in items]
    p_bar = sum(
        (sum(count * count for count in row) - n) / (n * (n - 1))
        for row in counts
    ) / len(counts)
    totals = [sum(row[index] for row in counts) for index in range(len(categories))]
    proportions = [total / (len(items) * n) for total in totals]
    expected = sum(value * value for value in proportions)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(p_bar, 1.0) else 0.0
    return (p_bar - expected) / (1.0 - expected)


def _expected_population() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for event in read_jsonl(PILOT_EVENTS):
        event_id = str(event["event_id"])
        graph = load_graph(str(event["instruction"]))
        result[event_id] = {
            "constraint_graph_sha256": graph.canonical_sha256(),
            "constraint_ids": [item.constraint_id for item in graph.constraints],
            "steps": list(range(int(event["prefix_start"]), int(event["prefix_end"]) + 1)),
        }
    if len(result) != 300:
        raise FormalAuditError("formal population must contain all 300 frozen events")
    return result


def _review_values(
    row: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, str]:
    required = {
        "schema_version", "reviewer_id", "reviewer_blinded_to_outcomes",
        "reviewer_blinded_to_qwen_factors", "event_id",
        "constraint_graph_sha256", "constraint_reviews", "review_complete",
    }
    if set(row) != required or row.get("schema_version") != SCHEMA:
        raise FormalAuditError("formal review row schema mismatch")
    if row.get("reviewer_blinded_to_outcomes") is not True or row.get("reviewer_blinded_to_qwen_factors") is not True:
        raise FormalAuditError("formal reviewer is not blinded")
    if row.get("review_complete") is not True:
        raise FormalAuditError("formal review row is incomplete")
    if row.get("constraint_graph_sha256") != expected["constraint_graph_sha256"]:
        raise FormalAuditError("formal review graph hash drift")
    reviews = row.get("constraint_reviews")
    constraint_ids = list(expected["constraint_ids"])
    steps = list(expected["steps"])
    if not isinstance(reviews, Mapping) or set(reviews) != set(constraint_ids):
        raise FormalAuditError("formal constraint population is incomplete")
    event_id = str(row["event_id"])
    values: dict[str, str] = {}
    for constraint_id in constraint_ids:
        item = reviews[constraint_id]
        if not isinstance(item, Mapping) or set(item) != {"dec_role", "factor_by_step"}:
            raise FormalAuditError("formal constraint item schema mismatch")
        role = DecRole(item["dec_role"]).value
        values[f"{event_id}::{constraint_id}::DEC_ROLE"] = role
        factors = item["factor_by_step"]
        if not isinstance(factors, list) or len(factors) != len(steps):
            raise FormalAuditError("formal factor population is incomplete")
        vectors = ([], [], [])
        for expected_step, factor in zip(steps, factors, strict=True):
            if not isinstance(factor, Mapping) or set(factor) != {
                "step", "instantiated", "distinguishable", "resolved"
            } or factor["step"] != expected_step:
                raise FormalAuditError("formal factor row schema/order mismatch")
            raw = [factor["instantiated"], factor["distinguishable"], factor["resolved"]]
            if any(type(value) is not bool for value in raw):
                raise FormalAuditError("formal factor labels must be boolean")
            for letter, value, vector in zip(("S", "G", "E"), raw, vectors, strict=True):
                label = "1" if value else "0"
                values[f"{event_id}::{constraint_id}::{expected_step}::{letter}"] = label
                vector.append(value)
        uad = derive_constraint_uad(*vectors, stability_k=3)[-1].value
        values[f"{event_id}::{constraint_id}::UAD"] = uad
    return values


def load_formal_reviews(
    paths: Sequence[Path],
    expected_population: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, str]], list[str]]:
    if len(paths) != 3:
        raise FormalAuditError("exactly three independent reviewer files are required")
    reviewer_ids: list[str] = []
    outputs: list[dict[str, str]] = []
    expected_ids = set(expected_population)
    for path in paths:
        rows = read_jsonl(path)
        by_event = {str(row.get("event_id")): row for row in rows}
        if len(rows) != len(by_event) or set(by_event) != expected_ids:
            raise FormalAuditError("reviewer file does not cover the exact formal population")
        ids = {str(row.get("reviewer_id", "")).strip() for row in rows}
        if len(ids) != 1 or not next(iter(ids)):
            raise FormalAuditError("one nonempty reviewer ID is required per file")
        reviewer_ids.append(next(iter(ids)))
        values: dict[str, str] = {}
        for event_id in sorted(expected_ids):
            values.update(_review_values(by_event[event_id], expected_population[event_id]))
        outputs.append(values)
    if len(set(reviewer_ids)) != 3:
        raise FormalAuditError("the three reviewer identities must be distinct")
    if any(set(values) != set(outputs[0]) for values in outputs[1:]):
        raise FormalAuditError("reviewer item populations differ")
    return outputs, reviewer_ids


def _item_kind(item_id: str) -> str:
    suffix = item_id.rsplit("::", 1)[-1]
    return suffix if suffix in {"DEC_ROLE", "UAD", "S", "G", "E"} else "UNKNOWN"


def disagreement_items(
    reviews: Sequence[Mapping[str, str]],
    *,
    include_derived_uad: bool = False,
) -> dict[str, list[str]]:
    return {
        item_id: [str(review[item_id]) for review in reviews]
        for item_id in sorted(reviews[0])
        if len({review[item_id] for review in reviews}) > 1
        and (include_derived_uad or _item_kind(item_id) != "UAD")
    }


def _load_adjudication(
    path: Path | None,
    *,
    disagreements: Mapping[str, Sequence[str]],
    reviewer_ids: Sequence[str],
) -> tuple[str | None, dict[str, str] | None]:
    if path is None:
        return None, None
    value = read_json(path)
    required = {
        "schema_version", "adjudicator_id", "adjudicator_blinded_to_outcomes",
        "items", "adjudication_complete",
    }
    if set(value) != required or value.get("schema_version") != ADJUDICATION_SCHEMA:
        raise FormalAuditError("adjudication schema mismatch")
    adjudicator = str(value.get("adjudicator_id", "")).strip()
    if not adjudicator or adjudicator in set(reviewer_ids):
        raise FormalAuditError("adjudicator must be one distinct person")
    if value.get("adjudicator_blinded_to_outcomes") is not True:
        raise FormalAuditError("adjudicator is not outcome-blinded")
    items = value.get("items")
    if value.get("adjudication_complete") is not True or not isinstance(items, Mapping):
        return adjudicator, None
    if set(items) != set(disagreements):
        return adjudicator, None
    allowed = {
        "DEC_ROLE": {role.value for role in DecRole},
        "UAD": {"U", "A", "D"},
        "S": {"0", "1"}, "G": {"0", "1"}, "E": {"0", "1"},
    }
    normalized = {str(key): str(label) for key, label in items.items()}
    for item_id, label in normalized.items():
        if label not in allowed[_item_kind(item_id)]:
            raise FormalAuditError(f"invalid adjudicated label: {item_id}")
    return adjudicator, normalized


def audit_formal_reviews(
    review_paths: Sequence[Path],
    *,
    expected_population: Mapping[str, Mapping[str, object]],
    adjudication_path: Path | None = None,
    gold_path: Path | None = None,
) -> dict[str, object]:
    reviews, reviewers = load_formal_reviews(review_paths, expected_population)
    uad_items = [
        [review[item_id] for review in reviews]
        for item_id in sorted(reviews[0]) if _item_kind(item_id) == "UAD"
    ]
    evidence_items = [
        [review[item_id] for review in reviews]
        for item_id in sorted(reviews[0]) if _item_kind(item_id) == "E"
    ]
    uad_kappa = fleiss_kappa(uad_items, ("U", "A", "D"))
    evidence_kappa = fleiss_kappa(evidence_items, ("0", "1"))
    kappa_pass = uad_kappa >= 0.65 and evidence_kappa >= 0.70
    disagreements = disagreement_items(reviews)
    derived_uad_disagreements = disagreement_items(
        reviews, include_derived_uad=True
    )
    derived_uad_disagreements = {
        key: value for key, value in derived_uad_disagreements.items()
        if _item_kind(key) == "UAD"
    }
    adjudicator, adjudicated = _load_adjudication(
        adjudication_path, disagreements=disagreements, reviewer_ids=reviewers
    )
    if not kappa_pass:
        status = "MF3ZP_LABEL_VALIDITY_FAIL"
    elif adjudicated is None:
        status = "MF3ZP_LABEL_VALIDITY_PENDING_ADJUDICATION"
    else:
        status = "MF3ZP_LABEL_VALIDITY_PASS"

    gold_inventory = None
    if status == "MF3ZP_LABEL_VALIDITY_PASS":
        if gold_path is None:
            raise FormalAuditError("immutable gold output path is required for PASS")
        final_labels: dict[str, str] = {}
        for item_id in sorted(reviews[0]):
            if _item_kind(item_id) == "UAD":
                continue
            votes = [review[item_id] for review in reviews]
            final_labels[item_id] = (
                votes[0] if len(set(votes)) == 1 else adjudicated[item_id]
            )
        for item_id in sorted(reviews[0]):
            if _item_kind(item_id) != "UAD":
                continue
            prefix = item_id.removesuffix("::UAD")
            factor_keys = {
                letter: sorted(
                    (
                        key for key in final_labels
                        if key.startswith(prefix + "::") and key.endswith("::" + letter)
                    ),
                    key=lambda key: int(key.rsplit("::", 2)[1]),
                )
                for letter in ("S", "G", "E")
            }
            if not factor_keys["S"] or not (
                len(factor_keys["S"]) == len(factor_keys["G"]) == len(factor_keys["E"])
            ):
                raise FormalAuditError("cannot derive adjudicated UAD from final factors")
            vectors = tuple(
                [final_labels[key] == "1" for key in factor_keys[letter]]
                for letter in ("S", "G", "E")
            )
            final_labels[item_id] = derive_constraint_uad(
                *vectors, stability_k=3
            )[-1].value
        gold_rows = [
            {"schema_version": GOLD_SCHEMA, "item_id": key, "label": final_labels[key]}
            for key in sorted(final_labels)
        ]
        atomic_jsonl(gold_path, gold_rows)
        gold_inventory = inventory(gold_path)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "reviewer_count": 3,
        "reviewer_id_hashes": [hashlib.sha256(value.encode()).hexdigest() for value in sorted(reviewers)],
        "adjudicator_count": 1 if adjudicator is not None else 0,
        "adjudicator_distinct": adjudicator is not None and adjudicator not in reviewers,
        "events": len(expected_population),
        "uad_fleiss_kappa": uad_kappa,
        "evidence_closure_fleiss_kappa": evidence_kappa,
        "thresholds": {"uad": 0.65, "evidence_closure": 0.70},
        "non_unanimous_items": len(disagreements),
        "non_unanimous_derived_uad_items": len(derived_uad_disagreements),
        "adjudicated_items": len(adjudicated or {}),
        "adjudication_complete": adjudicated is not None,
        "gold": gold_inventory,
        "oracle_headroom_authorized": status == "MF3ZP_LABEL_VALIDITY_PASS",
        "checkpoint_generated": False,
        "public_split_access": PUBLIC_CLOSED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, nargs=3, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.result, args.gold):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FormalAuditError(f"refusing to overwrite: {path}")
    verify_scout_protocol()
    result = audit_formal_reviews(
        args.reviews,
        expected_population=_expected_population(),
        adjudication_path=args.adjudication,
        gold_path=args.gold,
    )
    atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "MF3ZP_LABEL_VALIDITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
