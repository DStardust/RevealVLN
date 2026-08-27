#!/usr/bin/env python3
"""Repair only overlong display text in expansion branch proposals.

The provider response is retained byte-for-byte in the source attempt.  This
script may shorten ``visual_descriptor`` and ``rationale`` strings, but it
never changes branch identity, direction, evidence, target resolution,
confidence, flags, or any other semantic field.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import run_phase0c_cr5_multiview_branch as contract
import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
INPUT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/multiview_factory/"
    "RXR_PRIMARY_MULTIVIEW_INPUTS.json")
RESULT_DIR = ROOT / (
    "artifacts/phase1/rxr_train_expansion/branch_factory/results")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def shorten(value: str, limit: int) -> str:
    return value[:limit].rstrip()


def main() -> int:
    if not INPUT.is_file() or INPUT.is_symlink():
        raise SystemExit("multiview input is not ready")
    input_sha = sha256_file(INPUT)
    document = json.loads(INPUT.read_text())
    if (document.get("status") != "READY_FOR_BRANCH_PROPOSER"
            or document.get("training_authorized") is not False):
        raise SystemExit("multiview input contract failure")
    events = {row["event_id"]: row for row in document["events"]}
    repaired = []
    skipped = []
    for event_id, event in sorted(events.items()):
        directory = factory.result_directory(event)
        paths = sorted(directory.glob("attempt_*.json"), reverse=True)
        if not paths:
            continue
        source_path = paths[0]
        source = json.loads(source_path.read_text())
        if source.get("status") != "INVALID_MLLM_PROPOSAL":
            continue
        errors = source.get("validation_errors")
        if (not isinstance(errors, list) or not errors
                or any(error != "rationale" and not (
                    error.startswith("branch[")
                    and error.endswith("]:descriptor"))
                    for error in errors)):
            skipped.append({"event_id": event_id,
                            "reason": "NON_TEXT_VALIDATION_ERROR",
                            "validation_errors": errors})
            continue
        proposal = json.loads(json.dumps(source["normalized_proposal"]))
        changes = []
        repairable = True
        for index, branch in enumerate(proposal.get("branches", [])):
            descriptor = branch.get("visual_descriptor")
            if isinstance(descriptor, str) and len(descriptor) > 180:
                shortened = shorten(descriptor, 180)
                if len(shortened) < 3:
                    repairable = False
                    break
                branch["visual_descriptor"] = shortened
                changes.append({
                    "field": "branches[%d].visual_descriptor" % index,
                    "rule": "deterministic_right_truncation_to_180_chars",
                    "before_length": len(descriptor),
                    "after_length": len(shortened),
                })
        rationale = proposal.get("rationale")
        if isinstance(rationale, str) and len(rationale) > 600:
            shortened = shorten(rationale, 600)
            if not shortened:
                repairable = False
            else:
                proposal["rationale"] = shortened
                changes.append({
                    "field": "rationale",
                    "rule": "deterministic_right_truncation_to_600_chars",
                    "before_length": len(rationale),
                    "after_length": len(shortened),
                })
        adapted = factory.contract_event(event)
        remaining = contract.validate_proposal(proposal, adapted)
        if not repairable or not changes or remaining:
            skipped.append({"event_id": event_id,
                            "reason": "TEXT_REPAIR_NOT_SUFFICIENT",
                            "validation_errors": remaining})
            continue
        repaired_value = json.loads(json.dumps(source))
        repaired_value["status"] = "VALID_MLLM_PROPOSAL"
        repaired_value["normalized_proposal"] = proposal
        repaired_value["normalizations"] = list(
            repaired_value.get("normalizations", [])) + changes
        repaired_value["validation_errors"] = []
        repaired_value["deterministic_text_repair"] = {
            "revision": "rxr-multiview-display-text-repair/1",
            "source_path": str(source_path.relative_to(ROOT)),
            "source_sha256": sha256_file(source_path),
            "input_sha256": input_sha,
            "changes": changes,
            "display_text_content_changed": True,
            "operational_label_fields_changed": False,
            "semantic_equivalence_of_truncated_text_claimed": False,
            "provider_called": False,
        }
        output_path = factory.next_attempt(event)
        atomic_json(output_path, repaired_value)
        repaired.append({
            "event_id": event_id,
            "source": str(source_path.relative_to(ROOT)),
            "output": str(output_path.relative_to(ROOT)),
            "sha256": sha256_file(output_path),
            "changes": changes,
        })
    summary = {
        "status": "PASS",
        "revision": "rxr-multiview-display-text-repair-run/1",
        "input_sha256": input_sha,
        "repaired_count": len(repaired),
        "skipped_nonrepairable_count": len(skipped),
        "repaired": repaired,
        "skipped_nonrepairable": skipped,
        "display_text_content_changed": bool(repaired),
        "operational_label_fields_changed": False,
        "semantic_equivalence_of_truncated_text_claimed": False,
        "provider_calls_made": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    path = RESULT_DIR.parent / "RXR_MULTIVIEW_BRANCH_TEXT_REPAIR.json"
    atomic_json(path, summary)
    print(json.dumps({
        "status": summary["status"],
        "repaired": len(repaired),
        "skipped_nonrepairable": len(skipped),
        "output": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
