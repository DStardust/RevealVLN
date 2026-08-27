#!/usr/bin/env python3
"""Repair only overlong display text in failed hindsight responses."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import run_rxr_hindsight_event_factory as factory


ROOT = Path("/mnt/daiyang/vla")
RESULT_DIR = ROOT / (
    "artifacts/phase1/rxr_train_expansion/hindsight_factory/results")
OUT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/hindsight_factory/"
    "RXR_HINDSIGHT_EVENT_FACTORY_REPAIRS.json")
REPAIRABLE = re.compile(
    r"interval\[([0-9]+)\]:(reference_route_choice_summary|rationale)")
LIMITS = {"reference_route_choice_summary": 240, "rationale": 600}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def has_valid(directory: Path) -> bool:
    for path in directory.glob("attempt_*.json"):
        try:
            if json.loads(path.read_text())["status"] in {
                    "VALID_MLLM_PROPOSAL", "FACTORY_INPUT_FAILURE"}:
                return True
        except (KeyError, OSError, json.JSONDecodeError):
            pass
    return False


def main() -> int:
    repairs_by_destination = {}
    for path in sorted(RESULT_DIR.glob("order*_ep*/attempt_*.json")):
        try:
            value = json.loads(path.read_text())
            changes = value.get("posthoc_repairs")
            source = value.get("repair_source")
            if (value.get("status") == "VALID_MLLM_PROPOSAL"
                    and isinstance(changes, list) and changes
                    and isinstance(source, dict)):
                destination = str(path.relative_to(ROOT))
                repairs_by_destination[destination] = {
                    "source": source,
                    "destination": destination,
                    "destination_sha256": sha256_file(path),
                    "changes": changes,
                }
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    new_repair_count = 0
    unrepairable = []
    for directory in sorted(RESULT_DIR.glob("order*_ep*")):
        attempts = sorted(directory.glob("attempt_*.json"))
        if not attempts or has_valid(directory):
            continue
        source_path = attempts[-1]
        source = json.loads(source_path.read_text())
        errors = source.get("validation_errors", [])
        matches = [REPAIRABLE.fullmatch(value) for value in errors]
        if (source.get("status") != "INVALID_MLLM_PROPOSAL"
                or not errors or not all(matches)):
            unrepairable.append({
                "directory": str(directory.relative_to(ROOT)),
                "latest_attempt": str(source_path.relative_to(ROOT)),
                "latest_attempt_sha256": sha256_file(source_path),
                "errors": errors or [source.get("error_type", "UNKNOWN")],
            })
            continue
        normalized = copy.deepcopy(source["normalized_proposal"])
        changes = []
        for match in matches:
            index, field = int(match.group(1)), match.group(2)
            value = normalized["candidate_intervals"][index][field]
            limit = LIMITS[field]
            if not isinstance(value, str) or len(value) <= limit:
                raise SystemExit("repair preimage does not exceed limit")
            repaired = value[:limit]
            normalized["candidate_intervals"][index][field] = repaired
            changes.append({
                "field": "candidate_intervals[%d].%s" % (index, field),
                "rule": "unicode_codepoint_prefix_truncation_to_schema_max",
                "semantic_interval_fields_changed": False,
                "original_length": len(value),
                "repaired_length": len(repaired),
                "original_sha256": hashlib.sha256(
                    value.encode("utf-8")).hexdigest(),
                "repaired_sha256": hashlib.sha256(
                    repaired.encode("utf-8")).hexdigest(),
            })
        request = source["request_evidence"]
        record = {
            "trajectory_id": request["trajectory_id"],
            "timeline_frame_ids": request["timeline_frame_ids"],
            "deterministic_segments": request["deterministic_segments"],
        }
        post_errors = factory.validate(normalized, record)
        if post_errors:
            raise SystemExit("post-repair validation failed: "
                             + repr(post_errors))
        repaired = copy.deepcopy(source)
        repaired["status"] = "VALID_MLLM_PROPOSAL"
        repaired["normalized_proposal"] = normalized
        repaired["validation_errors"] = []
        repaired["posthoc_repairs"] = changes
        repaired["repair_source"] = {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": sha256_file(source_path),
        }
        repaired["repair_scope"] = (
            "schema-bounded display text only; interval, frame, clause, kind, "
            "pattern and confidence fields unchanged; free-text content is "
            "truncated and no semantic-equivalence claim is made")
        destination = directory / ("attempt_%03d.json" % (len(attempts) + 1))
        atomic_json(destination, repaired)
        destination_name = str(destination.relative_to(ROOT))
        repairs_by_destination[destination_name] = {
            "source": repaired["repair_source"],
            "destination": destination_name,
            "destination_sha256": sha256_file(destination),
            "changes": changes,
        }
        new_repair_count += 1
    repairs = [repairs_by_destination[key]
               for key in sorted(repairs_by_destination)]
    output = {
        "manifest": "RxR hindsight event factory bounded-text repairs",
        "revision": "rxr-hindsight-event-factory-repairs/1",
        "status": "COMPLETE",
        "repair_count": len(repairs),
        "new_repair_count": new_repair_count,
        "unrepairable_count": len(unrepairable),
        "repairs": repairs,
        "unrepairable": unrepairable,
        "repair_interpretation": {
            "display_text_content_changed": bool(repairs),
            "operational_label_fields_changed": False,
            "semantic_equivalence_of_truncated_text_claimed": False,
            "legacy_attempt_scope_text_is_superseded_by_this_record": True,
        },
        "network_calls_made": 0,
        "geometry_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "repairs": len(repairs),
        "new_repairs": new_repair_count,
        "unrepairable": len(unrepairable),
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
