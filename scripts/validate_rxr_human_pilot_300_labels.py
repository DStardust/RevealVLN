#!/usr/bin/env python3
"""Fail-closed validation of the single-reviewer RxR 300 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
PACKAGE = ROOT / "artifacts/phase1/rxr_train_expansion/human_pilot_300"
SELECTION = PACKAGE / "RXR_HUMAN_PILOT_300_SELECTION.json"
DEFAULT_INPUT = PACKAGE / "daiyang_rxr300.jsonl"
DEFAULT_OUTPUT = PACKAGE / "RXR_HUMAN_PILOT_300_LABEL_ACCEPTANCE.json"
EXPECTED_SELECTION_SHA256 = (
    "c8d79f8aa7285b050568759d5f492ff288186520551749e6a5f10a2c55dfd179"
)
CHECKS = (
    "two_distinct_executable_exits",
    "alternative_is_not_incoming_closed_or_duplicate",
    "instruction_uniquely_selects_target",
    "decision_center_and_temporal_order_are_reasonable",
    "causal_prefix_supports_reveal_without_future_frames",
)
REASON_FOR_CHECK = {
    CHECKS[0]: "NO_TWO_DISTINCT_EXECUTABLE_EXITS",
    CHECKS[1]: "ALTERNATIVE_INCOMING_CLOSED_DUPLICATE_OR_SHORT",
    CHECKS[2]: "INSTRUCTION_TARGET_NOT_UNIQUE",
    CHECKS[3]: "DECISION_CENTER_OR_TEMPORAL_ORDER_INVALID",
    CHECKS[4]: "CAUSAL_REVEAL_NEEDS_FUTURE_OR_IS_NOT_SUPPORTED",
}
ALLOWED_REASONS = set(REASON_FOR_CHECK.values()) | {
    "INSUFFICIENT_VISUAL_EVIDENCE"
}
REQUIRED_KEYS = {
    "reviewer_id", "reviewer_type", "event_id", *CHECKS,
    "final_label", "reason_codes", "comment_zh",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def wilson_interval(successes: int, total: int, z: float = 1.95996398454):
    if total == 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [round(max(0.0, center - radius), 6),
            round(min(1.0, center + radius), 6)]


def parse_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {number}: row is not an object")
            rows.append((number, value))
    return rows


def validate_row(number: int, row: dict) -> list[str]:
    errors = []
    if set(row) != REQUIRED_KEYS:
        errors.append(
            f"line {number}: key set mismatch; missing="
            f"{sorted(REQUIRED_KEYS - set(row))}, extra="
            f"{sorted(set(row) - REQUIRED_KEYS)}"
        )
    reviewer = row.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append(f"line {number}: reviewer_id must be nonempty")
    if row.get("reviewer_type") != "HUMAN":
        errors.append(f"line {number}: reviewer_type must equal HUMAN")
    if not isinstance(row.get("event_id"), str) or not row.get("event_id"):
        errors.append(f"line {number}: event_id must be nonempty")
    comment = row.get("comment_zh")
    if not isinstance(comment, str):
        errors.append(f"line {number}: comment_zh must be a string")
    reasons = row.get("reason_codes")
    if (not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
            or len(reasons) != len(set(reasons))):
        errors.append(f"line {number}: reason_codes must be a unique string list")
        reasons = []
    unknown = set(reasons) - ALLOWED_REASONS
    if unknown:
        errors.append(f"line {number}: unknown reason codes {sorted(unknown)}")
    label = row.get("final_label")
    values = [row.get(key) for key in CHECKS]
    if any(value not in (True, False, None) for value in values):
        errors.append(f"line {number}: review checks must be true/false/null")
    if label == "ACCEPT":
        if values != [True] * len(CHECKS) or reasons:
            errors.append(
                f"line {number}: ACCEPT requires five true checks and no reasons"
            )
    elif label == "REJECT":
        false_keys = [key for key in CHECKS if row.get(key) is False]
        expected_reasons = {REASON_FOR_CHECK[key] for key in false_keys}
        if not false_keys or not reasons or not expected_reasons <= set(reasons):
            errors.append(
                f"line {number}: REJECT needs a false check and its reason code"
            )
        if "INSUFFICIENT_VISUAL_EVIDENCE" in reasons:
            errors.append(
                f"line {number}: insufficient evidence belongs to AMBIGUOUS"
            )
    elif label == "AMBIGUOUS":
        if (values != [None] * len(CHECKS)
                or reasons != ["INSUFFICIENT_VISUAL_EVIDENCE"]):
            errors.append(
                f"line {number}: AMBIGUOUS requires five null checks and its sole reason"
            )
    else:
        errors.append(f"line {number}: invalid final_label {label!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    project = ROOT.resolve()
    if project not in source.parents or project not in output.parents:
        raise SystemExit("input and output must resolve inside the project")
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"missing regular non-symlink review file: {source}")
    if sha256_file(SELECTION) != EXPECTED_SELECTION_SHA256:
        raise SystemExit("frozen 300-item selection drift")

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    items = selection["items"]
    expected_ids = [item["event_id"] for item in items]
    metadata = {item["event_id"]: item for item in items}
    parsed = parse_jsonl(source)
    failures = []
    for number, row in parsed:
        failures.extend(validate_row(number, row))
    rows = [row for _, row in parsed]
    observed_ids = [row.get("event_id") for row in rows]
    duplicates = sorted(
        event_id for event_id, count in Counter(observed_ids).items()
        if count > 1
    )
    if len(rows) != 300:
        failures.append(f"expected 300 nonblank rows, observed {len(rows)}")
    if duplicates:
        failures.append(f"duplicate event IDs: {duplicates[:10]}")
    if set(observed_ids) != set(expected_ids):
        failures.append(
            "event set mismatch: missing=%s extra=%s" % (
                sorted(set(expected_ids) - set(observed_ids))[:10],
                sorted(set(observed_ids) - set(expected_ids))[:10],
            )
        )
    if observed_ids != expected_ids:
        failures.append("JSONL row order differs from sealed review order")
    reviewer_ids = sorted({
        row.get("reviewer_id", "").strip()
        for row in rows if isinstance(row.get("reviewer_id"), str)
    })
    if len(reviewer_ids) != 1:
        failures.append(
            f"single-review pilot requires exactly one reviewer_id; got {reviewer_ids}"
        )

    by_cohort = defaultdict(Counter)
    scenes_by_label = defaultdict(set)
    for row in rows:
        event_id = row.get("event_id")
        if event_id not in metadata:
            continue
        cohort = metadata[event_id]["cohort"]
        label = row.get("final_label")
        by_cohort[cohort][label] += 1
        scenes_by_label[label].add(metadata[event_id]["scene_id"])
    core = by_cohort["AUDIT_CORE_UNIFORM_250"]
    core_total = sum(core.values())
    core_accept = core.get("ACCEPT", 0)
    all_counts = Counter(row.get("final_label") for row in rows)
    gates = {
        "selection_sha256_exact": sha256_file(SELECTION)
            == EXPECTED_SELECTION_SHA256,
        "exact_300_rows": len(rows) == 300,
        "exact_unique_event_set": (
            not duplicates and set(observed_ids) == set(expected_ids)
        ),
        "sealed_order_exact": observed_ids == expected_ids,
        "schema_and_label_semantics_valid": not any(
            failure.startswith("line ") for failure in failures
        ),
        "one_human_reviewer": len(reviewer_ids) == 1,
        "all_rows_decided": sum(all_counts.values()) == 300
            and set(all_counts) <= {"ACCEPT", "REJECT", "AMBIGUOUS"},
    }
    result = {
        "manifest": "RevealNav RxR human pilot 300 label acceptance",
        "revision": "rxr-human-pilot-300-label-acceptance/1",
        "status": "HUMAN_LABELS_PASS_TX_JOIN_REQUIRED"
            if all(gates.values()) else "HUMAN_LABELS_FAIL",
        "scope": (
            "single-reviewer construction audit; not three-reviewer agreement "
            "and not final benchmark Gold"
        ),
        "sources": {
            str(SELECTION.relative_to(ROOT)): sha256_file(SELECTION),
            str(source.relative_to(project)): sha256_file(source),
        },
        "reviewer_count": len(reviewer_ids),
        "reviewer_ids": reviewer_ids,
        "counts": {
            "rows": len(rows),
            "unique_events": len(set(observed_ids)),
            "labels": dict(sorted(all_counts.items(), key=lambda item: str(item[0]))),
            "scenes_by_label": {
                str(label): len(scenes) for label, scenes in scenes_by_label.items()
            },
            "by_cohort": {
                cohort: dict(sorted(counts.items(), key=lambda item: str(item[0])))
                for cohort, counts in by_cohort.items()
            },
        },
        "primary_unbiased_core_estimate": {
            "cohort": "AUDIT_CORE_UNIFORM_250",
            "accepts": core_accept,
            "total": core_total,
            "accept_rate": round(core_accept / core_total, 6)
                if core_total else None,
            "wilson_95pct": wilson_interval(core_accept, core_total),
            "supplement_excluded_from_estimate": True,
        },
        "gates": gates,
        "failures": failures,
        "three_reviewer_agreement_measured": False,
        "human_labels_fabricated": 0,
        "training_authorized": False,
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"], "counts": result["counts"],
        "core": result["primary_unbiased_core_estimate"],
        "failures": failures[:20],
        "output": str(output.relative_to(project)),
    }, indent=2, ensure_ascii=False))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
