#!/usr/bin/env python3
"""Validate the fresh 100-event audit and authorize filtered head training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
AUDIT = BASE / "multibranch_fullset_audit_100"
V2 = BASE / "multibranch_v2"
SELECTION = AUDIT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_SELECTION.json"
PACKAGE_MANIFEST = AUDIT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_MANIFEST.json"
PACKAGE_ACCEPTANCE = AUDIT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_PACKAGE_ACCEPTANCE.json"
LABELS = AUDIT / "daiyang_fullset100.jsonl"
LABEL_ACCEPTANCE = AUDIT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_LABEL_ACCEPTANCE.json"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
TX_GATE = V2 / "RXR_MULTIBRANCH_TX_V2_GATE.json"
FEATURE_GATE = V2 / "RXR_MULTIBRANCH_FEATURE_GATE_V2.json"
FEATURE_MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2.json"
AUTHORIZED_MANIFEST = (
    V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
)
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"

CHECKS = (
    "candidate_set_complete",
    "all_candidates_distinct_and_executable",
    "instruction_uniquely_selects_target_among_all",
    "decision_center_and_temporal_order_reasonable",
    "causal_prefix_supports_reveal_without_future_frames",
)
REASON_FOR_CHECK = {
    CHECKS[0]: "CANDIDATE_SET_INCOMPLETE",
    CHECKS[1]: "CANDIDATE_INCOMING_CLOSED_DUPLICATE_OR_SHORT",
    CHECKS[2]: "INSTRUCTION_TARGET_NOT_UNIQUE_AMONG_ALL",
    CHECKS[3]: "DECISION_CENTER_OR_TEMPORAL_ORDER_INVALID",
    CHECKS[4]: "CAUSAL_REVEAL_NEEDS_FUTURE_OR_IS_NOT_SUPPORTED",
}
AMBIGUOUS_REASON = "INSUFFICIENT_VISUAL_EVIDENCE"
REQUIRED_KEYS = {
    "reviewer_id", "reviewer_type", "event_id", *CHECKS,
    "final_label", "reason_codes", "comment_zh",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(part, path)


def load_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
        raise RuntimeError(f"missing or unsafe input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
        raise RuntimeError(f"missing or unsafe label input: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"line {number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"line {number}: row is not an object")
        rows.append(value)
    return rows


def row_errors(row: dict, number: int) -> list[str]:
    errors = []
    if set(row) != REQUIRED_KEYS:
        errors.append(f"line {number}: key set mismatch")
    if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip():
        errors.append(f"line {number}: reviewer_id must be nonempty")
    if row.get("reviewer_type") != "HUMAN":
        errors.append(f"line {number}: reviewer_type must equal HUMAN")
    if not isinstance(row.get("event_id"), str) or not row["event_id"]:
        errors.append(f"line {number}: event_id must be nonempty")
    if not isinstance(row.get("comment_zh"), str):
        errors.append(f"line {number}: comment_zh must be a string")
    values = [row.get(key) for key in CHECKS]
    if any(value not in (True, False, None) for value in values):
        errors.append(f"line {number}: checks must be true, false, or null")
    reasons = row.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        errors.append(f"line {number}: reason_codes must be a unique string list")
        reasons = []
    allowed = set(REASON_FOR_CHECK.values()) | {AMBIGUOUS_REASON}
    if set(reasons) - allowed:
        errors.append(f"line {number}: unknown reason code")
    label = row.get("final_label")
    if label == "ACCEPT":
        if values != [True] * len(CHECKS) or reasons:
            errors.append(f"line {number}: ACCEPT semantics mismatch")
    elif label == "REJECT":
        false_checks = [key for key in CHECKS if row.get(key) is False]
        expected = {REASON_FOR_CHECK[key] for key in false_checks}
        if not false_checks or set(reasons) != expected:
            errors.append(f"line {number}: REJECT semantics mismatch")
    elif label == "AMBIGUOUS":
        if values != [None] * len(CHECKS) or reasons != [AMBIGUOUS_REASON]:
            errors.append(f"line {number}: AMBIGUOUS semantics mismatch")
    else:
        errors.append(f"line {number}: invalid final_label")
    return errors


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.95996398454
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total
        + z * z / (4 * total * total)
    ) / denominator
    return [round(center - radius, 6), round(center + radius, 6)]


def main() -> int:
    selection = load_json(SELECTION)
    package_manifest = load_json(PACKAGE_MANIFEST)
    package_acceptance = load_json(PACKAGE_ACCEPTANCE)
    index = load_json(INDEX)
    tx_gate = load_json(TX_GATE)
    feature_gate = load_json(FEATURE_GATE)
    feature_manifest = load_json(FEATURE_MANIFEST)
    rows = load_jsonl(LABELS)

    expected_ids = [item["event_id"] for item in package_manifest["items"]]
    selected_ids = [item["event_id"] for item in selection["items"]]
    observed_ids = [row.get("event_id") for row in rows]
    failures = []
    for number, row in enumerate(rows, 1):
        failures.extend(row_errors(row, number))
    if len(rows) != 100:
        failures.append(f"expected 100 rows, observed {len(rows)}")
    if len(set(observed_ids)) != len(observed_ids):
        failures.append("duplicate event IDs")
    if observed_ids != expected_ids or expected_ids != selected_ids:
        failures.append("label, package, and sealed selection order mismatch")
    reviewers = sorted({
        row.get("reviewer_id", "").strip()
        for row in rows if isinstance(row.get("reviewer_id"), str)
    })
    if len(reviewers) != 1:
        failures.append("exactly one human reviewer is required")
    package_gates = {
        "package_acceptance_pass": package_acceptance.get("status")
        == "PACKAGE_PASS_READY_FOR_HUMAN_REVIEW",
        "selection_was_frozen_before_labels": selection.get("status")
        == "SELECTION_FROZEN_BEFORE_HUMAN_LABELING",
        "selection_training_was_not_pre_authorized": selection.get(
            "training_authorized"
        ) is False,
        "selection_sha256_bound": package_acceptance.get("selection_sha256")
        == sha256_file(SELECTION),
        "package_manifest_sha256_bound": package_acceptance.get("manifest_sha256")
        == sha256_file(PACKAGE_MANIFEST),
        "upstream_tx_pass": tx_gate.get("status") == "MULTIBRANCH_TX_PASS",
        "upstream_feature_gate_pass": feature_gate.get("status")
        == "FEATURE_GATE_PASS_AUDIT_REQUIRED",
        "upstream_feature_manifest_bound": feature_gate.get("manifest", {}).get(
            "sha256"
        ) == sha256_file(FEATURE_MANIFEST),
    }
    for name, passed in package_gates.items():
        if not passed:
            failures.append(f"precondition failed: {name}")

    labels = Counter(row.get("final_label") for row in rows)
    item_by_id = {item["event_id"]: item for item in package_manifest["items"]}
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_branch_count: dict[str, Counter] = defaultdict(Counter)
    reasons = Counter()
    for row in rows:
        item = item_by_id.get(row.get("event_id"))
        if item is None:
            continue
        by_split[item["split"]][row["final_label"]] += 1
        by_branch_count[str(item["candidate_branch_count"])][
            row["final_label"]
        ] += 1
        reasons.update(row["reason_codes"])
    accepts = labels["ACCEPT"]
    interval = wilson_interval(accepts, len(rows)) if rows else [0.0, 0.0]
    quality_gates = {
        "all_rows_decided_without_ambiguity": labels["AMBIGUOUS"] == 0,
        "audit_accept_fraction_at_least_90pct": accepts / len(rows) >= 0.90
        if rows else False,
        "audit_accept_wilson_95pct_lower_at_least_90pct": interval[0] >= 0.90,
        "all_31_three_branch_events_accepted": by_branch_count["3"]["ACCEPT"]
        == 31 and sum(by_branch_count["3"].values()) == 31,
    }
    for name, passed in quality_gates.items():
        if not passed:
            failures.append(f"quality gate failed: {name}")

    rejected_ids = [
        row["event_id"] for row in rows if row.get("final_label") == "REJECT"
    ]
    index_ids = [record["event_id"] for record in index.get("records", [])]
    feature_ids = [record["event_id"] for record in feature_manifest.get("records", [])]
    if len(index_ids) != len(set(index_ids)) or index_ids != feature_ids:
        failures.append("training index and feature manifest identities mismatch")
    if not set(rejected_ids) <= set(feature_ids):
        failures.append("one or more rejected events are absent from feature manifest")

    label_acceptance = {
        "schema_version": "revealnav-mf2-fullset-audit-label-acceptance/1",
        "status": "HUMAN_AUDIT_PASS" if not failures else "HUMAN_AUDIT_FAIL",
        "scope": (
            "single-human construction audit for exploratory head training; "
            "not three-reviewer agreement or a paper benchmark result"
        ),
        "sources": {
            str(SELECTION.relative_to(ROOT)): sha256_file(SELECTION),
            str(PACKAGE_MANIFEST.relative_to(ROOT)): sha256_file(PACKAGE_MANIFEST),
            str(PACKAGE_ACCEPTANCE.relative_to(ROOT)): sha256_file(PACKAGE_ACCEPTANCE),
            str(LABELS.relative_to(ROOT)): sha256_file(LABELS),
        },
        "reviewer_ids": reviewers,
        "counts": {
            "rows": len(rows),
            "labels": dict(sorted(labels.items())),
            "reasons": dict(sorted(reasons.items())),
            "by_split": {key: dict(sorted(value.items())) for key, value in sorted(by_split.items())},
            "by_candidate_branch_count": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_branch_count.items())
            },
        },
        "audit_accept_fraction": round(accepts / len(rows), 6) if rows else None,
        "audit_accept_wilson_95pct": interval,
        "package_gates": package_gates,
        "quality_gates": quality_gates,
        "quality_rule_provenance": (
            "post-return engineering safety gate; not preregistered and not "
            "eligible as a paper-level statistical claim"
        ),
        "rejected_event_ids": rejected_ids,
        "failures": failures,
        "three_reviewer_agreement_measured": False,
        "paper_benchmark_claim_authorized": False,
    }
    atomic_json(LABEL_ACCEPTANCE, label_acceptance)
    if failures:
        print(json.dumps(label_acceptance, indent=2, ensure_ascii=False))
        return 1

    kept_records = [
        record for record in feature_manifest["records"]
        if record["event_id"] not in set(rejected_ids)
    ]
    feature_failures = []
    for record in kept_records:
        path = (FEATURE_MANIFEST.parent / record["path"]).resolve()
        if (
            FEATURE_MANIFEST.parent.resolve() not in path.parents
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            feature_failures.append(record["event_id"])
    if feature_failures:
        label_acceptance["status"] = "HUMAN_AUDIT_FAIL"
        label_acceptance["failures"].append(
            "authorized feature provenance failure: " + ",".join(feature_failures[:10])
        )
        atomic_json(LABEL_ACCEPTANCE, label_acceptance)
        return 1

    split_counts = Counter(record["split"] for record in kept_records)
    authorized_manifest = {
        "schema_version": feature_manifest["schema_version"],
        "metadata": {
            **feature_manifest["metadata"],
            "training_authorized": True,
            "human_audit_status": "FRESH_FULLSET_AUDIT_PASS",
            "human_audit_labels_sha256": sha256_file(LABELS),
            "human_audit_accept_fraction": round(accepts / len(rows), 6),
            "known_human_rejects_excluded": len(rejected_ids),
            "authorized_event_count": len(kept_records),
            "paper_result": False,
        },
        "records": kept_records,
    }
    atomic_json(AUTHORIZED_MANIFEST, authorized_manifest)
    authorization = {
        "schema_version": "revealnav-mf2-training-authorization/2",
        "status": "TRAINING_AUTHORIZATION_PASS",
        "training_authorized": True,
        "scope": (
            "exploratory MF2 causal-head training on train split and model "
            "selection on development split; gold is not used for training"
        ),
        "training_manifest": {
            "path": str(AUTHORIZED_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(AUTHORIZED_MANIFEST),
        },
        "human_audit": {
            "acceptance_path": str(LABEL_ACCEPTANCE.relative_to(ROOT)),
            "acceptance_sha256": sha256_file(LABEL_ACCEPTANCE),
            "labels_path": str(LABELS.relative_to(ROOT)),
            "labels_sha256": sha256_file(LABELS),
            "accepts": accepts,
            "rejects": labels["REJECT"],
            "ambiguous": labels["AMBIGUOUS"],
            "accept_fraction": round(accepts / len(rows), 6),
            "wilson_95pct": interval,
        },
        "counts": {
            "source_events": len(feature_manifest["records"]),
            "authorized_events": len(kept_records),
            "excluded_human_rejects": len(rejected_ids),
            "authorized_by_split": dict(sorted(split_counts.items())),
        },
        "excluded_event_ids": rejected_ids,
        "known_rejects_absent_from_training_manifest": all(
            event_id not in {record["event_id"] for record in kept_records}
            for event_id in rejected_ids
        ),
        "future_frames_used_for_online_input": 0,
        "three_reviewer_agreement_measured": False,
        "paper_benchmark_claim_authorized": False,
        "full_submission_gate_satisfied": False,
    }
    atomic_json(AUTHORIZATION, authorization)
    print(json.dumps({
        "status": authorization["status"],
        "audit_labels": dict(sorted(labels.items())),
        "audit_accept_wilson_95pct": interval,
        "authorized_events": len(kept_records),
        "authorized_by_split": dict(sorted(split_counts.items())),
        "excluded_event_ids": rejected_ids,
        "training_manifest": str(AUTHORIZED_MANIFEST.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
