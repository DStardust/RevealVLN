#!/usr/bin/env python3
"""Aggregate three independent human review lanes without opening prior Gold."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1/new_gold"
PACKAGE = BASE / "review_package/RXR_NEW_GOLD_REVIEW_MANIFEST.json"
REVIEWS = BASE / "reviews"
OUT = BASE / "RXR_NEW_GOLD_HUMAN_CONSENSUS.json"
LANES = ("R1", "R2", "R3")
FIELDS = {
    "event_valid": {"ACCEPT", "REJECT", "AMBIGUOUS"},
    "q_state": {"U", "A", "D", "UNRESOLVABLE"},
    "target_in_set_at_q": {"YES", "NO", "AMBIGUOUS"},
    "candidate_separable_at_q": {"YES", "NO", "AMBIGUOUS"},
    "decisive_evidence_closed_at_q": {"YES", "NO", "AMBIGUOUS"},
    "multiple_executable_branches": {"YES", "NO", "AMBIGUOUS"},
}


def fleiss_kappa(rows: list[list[str]], categories: list[str]) -> float | None:
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(row) != n for row in rows):
        raise RuntimeError("invalid Fleiss matrix")
    counts = [Counter(row) for row in rows]
    p_bar = sum(
        (sum(value * value for value in count.values()) - n) / (n * (n - 1))
        for count in counts
    ) / len(counts)
    totals = Counter(value for row in rows for value in row)
    p_e = sum((totals[category] / (len(rows) * n)) ** 2 for category in categories)
    return None if math.isclose(1.0 - p_e, 0.0) else (p_bar - p_e) / (1.0 - p_e)


def majority(values: list[str]) -> str | None:
    if not values:
        return None
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count >= 2 else None


def main() -> int:
    package = json.loads(PACKAGE.read_text())
    expected = {row["event_id"] for row in package["items"]}
    by_lane = {}
    reviewer_ids = set()
    for lane in LANES:
        path = REVIEWS / f"{lane}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row.get("event_id") for row in rows]
        if len(rows) != len(expected) or set(ids) != expected or len(ids) != len(set(ids)):
            raise RuntimeError("review population mismatch: " + lane)
        lane_ids = {row.get("reviewer_id") for row in rows}
        if len(lane_ids) != 1 or None in lane_ids or "" in lane_ids:
            raise RuntimeError("reviewer identity mismatch: " + lane)
        reviewer_ids.update(lane_ids)
        for row in rows:
            if row.get("review_lane") != lane or row.get("reviewer_type") != "HUMAN":
                raise RuntimeError("review lane/type mismatch")
            if row.get("event_valid") not in FIELDS["event_valid"]:
                raise RuntimeError("incomplete event_valid")
            if row["event_valid"] == "ACCEPT" and any(
                row.get(field) not in allowed - {"AMBIGUOUS"}
                for field, allowed in FIELDS.items() if field != "event_valid"
            ):
                raise RuntimeError("accepted row has incomplete/ambiguous semantic fields")
        by_lane[lane] = {row["event_id"]: row for row in rows}
    if len(reviewer_ids) != 3:
        raise RuntimeError("the three review lanes must have distinct human reviewers")

    consensus = []
    uad_matrix = []
    closure_matrix = []
    for event_id in sorted(expected):
        reviews = [by_lane[lane][event_id] for lane in LANES]
        valid = majority([row["event_valid"] for row in reviews])
        row = {
            "event_id": event_id,
            "event_valid": valid,
            "q_state": majority([value["q_state"] for value in reviews if value.get("q_state")]),
            "target_in_set_at_q": majority([value["target_in_set_at_q"] for value in reviews if value.get("target_in_set_at_q")]),
            "candidate_separable_at_q": majority([value["candidate_separable_at_q"] for value in reviews if value.get("candidate_separable_at_q")]),
            "decisive_evidence_closed_at_q": majority([value["decisive_evidence_closed_at_q"] for value in reviews if value.get("decisive_evidence_closed_at_q")]),
            "multiple_executable_branches": majority([value["multiple_executable_branches"] for value in reviews if value.get("multiple_executable_branches")]),
            "reviewer_votes": {lane: by_lane[lane][event_id]["event_valid"] for lane in LANES},
        }
        row["semantic_consensus_complete"] = all(
            row[field] is not None for field in FIELDS
        )
        row["gold_semantic_accept"] = (
            row["event_valid"] == "ACCEPT"
            and row["multiple_executable_branches"] == "YES"
            and row["semantic_consensus_complete"]
        )
        if all(review.get("q_state") in FIELDS["q_state"] for review in reviews):
            uad_matrix.append([review["q_state"] for review in reviews])
        if all(review.get("decisive_evidence_closed_at_q") in FIELDS["decisive_evidence_closed_at_q"] for review in reviews):
            closure_matrix.append([review["decisive_evidence_closed_at_q"] for review in reviews])
        consensus.append(row)

    uad_kappa = fleiss_kappa(uad_matrix, sorted(FIELDS["q_state"]))
    closure_kappa = fleiss_kappa(closure_matrix, sorted(FIELDS["decisive_evidence_closed_at_q"]))
    accepted = sum(row["gold_semantic_accept"] for row in consensus)
    gates = {
        "three_distinct_human_reviewers": len(reviewer_ids) == 3,
        "at_least_600_semantic_accepts": accepted >= 600,
        "uad_fleiss_kappa_at_least_0_65": uad_kappa is not None and uad_kappa >= 0.65,
        "evidence_closure_kappa_at_least_0_70": closure_kappa is not None and closure_kappa >= 0.70,
    }
    output = {
        "schema_version": "revealnav-new-gold-human-consensus/1",
        "status": "HUMAN_GOLD_SEMANTICS_PASS_AUTOMATIC_CAUSAL_TX_REQUIRED" if all(gates.values()) else "HUMAN_GOLD_GATE_FAIL",
        "reviewer_ids": sorted(reviewer_ids),
        "counts": {"reviewed": len(consensus), "semantic_accepts": accepted},
        "agreement": {"uad_fleiss_kappa": uad_kappa, "evidence_closure_fleiss_kappa": closure_kappa},
        "gates": gates,
        "events": consensus,
        "old_gold_payload_read": False,
        "gold_authorized": False,
        "remaining_blocker": "automatic causal and resource-label closure on accepted events",
    }
    part = OUT.with_name(OUT.name + ".part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(part, OUT)
    print(json.dumps({"status": output["status"], "counts": output["counts"], "agreement": output["agreement"], "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
