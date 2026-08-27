#!/usr/bin/env python3
"""Seal first-response branch proposals and reject semantic invalids."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import run_phase0c_cr5_multiview_branch as contract
import run_rxr_multiview_branch_factory as factory


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
INPUT = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
OUT_DIR = BASE / "branch_factory"
REPAIR_DIR = OUT_DIR / "first_response_repairs"
ACCEPTED = OUT_DIR / "RXR_MULTIVIEW_BRANCH_ACCEPTED.json"
PRESCREEN = OUT_DIR / "RXR_MULTIVIEW_MACHINE_PRESCREEN.json"
HARD_FLAGS = {
    "no_alternative_exit",
    "retrace_only",
    "single_channel_turn",
    "target_language_ambiguous",
    "visual_evidence_ambiguous",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def text_only_repair(source: dict, event: dict) -> tuple[dict | None, list]:
    errors = source.get("validation_errors")
    if not isinstance(errors, list) or not errors or any(
            error != "rationale" and not (
                error.startswith("branch[") and error.endswith("]:descriptor")
            ) for error in errors):
        return None, []
    proposal = copy.deepcopy(source["normalized_proposal"])
    changes = []
    for index, branch in enumerate(proposal.get("branches", [])):
        value = branch.get("visual_descriptor")
        if isinstance(value, str) and len(value) > 180:
            shortened = value[:180].rstrip()
            if len(shortened) < 3:
                return None, []
            branch["visual_descriptor"] = shortened
            changes.append({
                "field": f"branches[{index}].visual_descriptor",
                "rule": "deterministic_right_truncation_to_180_chars",
                "before_length": len(value),
                "after_length": len(shortened),
            })
    rationale = proposal.get("rationale")
    if isinstance(rationale, str) and len(rationale) > 600:
        shortened = rationale[:600].rstrip()
        if not shortened:
            return None, []
        proposal["rationale"] = shortened
        changes.append({
            "field": "rationale",
            "rule": "deterministic_right_truncation_to_600_chars",
            "before_length": len(rationale),
            "after_length": len(shortened),
        })
    if not changes or contract.validate_proposal(
            proposal, factory.contract_event(event)):
        return None, []
    repaired = copy.deepcopy(source)
    repaired["status"] = "VALID_MLLM_PROPOSAL"
    repaired["normalized_proposal"] = proposal
    repaired["validation_errors"] = []
    repaired["normalizations"] = list(repaired.get("normalizations", [])) + changes
    repaired["deterministic_text_repair"] = {
        "revision": "rxr-first-response-display-text-repair/1",
        "source_attempt": "attempt_001.json",
        "changes": changes,
        "operational_label_fields_changed": False,
        "provider_called": False,
    }
    return repaired, changes


def main() -> int:
    manifest = json.loads(INPUT.read_text())
    if (manifest.get("status") != "READY_FOR_BRANCH_PROPOSER"
            or manifest.get("training_authorized") is not False):
        raise SystemExit("multiview input contract failure")
    input_sha = sha256_file(INPUT)
    accepted_rows = []
    rejected_rows = []
    accepted_payloads = {}
    for event in manifest["events"]:
        event_id = event["event_id"]
        source_path = factory.result_directory(event) / "attempt_001.json"
        if not source_path.is_file() or source_path.is_symlink():
            raise SystemExit("missing first response: " + event_id)
        source = json.loads(source_path.read_text())
        evidence = source.get("request_evidence", {})
        if (evidence.get("input_sha256") != input_sha
                or evidence.get("event_id") != event_id):
            raise SystemExit("first-response evidence drift: " + event_id)
        for media in evidence.get("media", []):
            factory.safe_media(media)

        payload = source
        payload_path = source_path
        if source.get("status") == "INVALID_MLLM_PROPOSAL":
            repaired, _ = text_only_repair(source, event)
            if repaired is not None:
                payload = repaired
                payload_path = REPAIR_DIR / (event_id + ".json")
                atomic_json(payload_path, payload)
        valid = (
            payload.get("status") == "VALID_MLLM_PROPOSAL"
            and payload.get("requested_model") == factory.MODEL
            and payload.get("provider_model") == factory.MODEL
            and payload.get("enable_thinking") is False
            and not contract.validate_proposal(
                payload["normalized_proposal"], factory.contract_event(event))
        )
        row = {
            "event_id": event_id,
            "expansion_order": event["expansion_order"],
            "episode_id": event["episode_id"],
            "proposal_path": str(payload_path.relative_to(ROOT)),
            "proposal_sha256": sha256_file(payload_path),
            "first_response_path": str(source_path.relative_to(ROOT)),
            "first_response_sha256": sha256_file(source_path),
            "human_reviewed": False,
            "training_label": False,
        }
        if valid:
            accepted_rows.append(row)
            accepted_payloads[event_id] = payload
        else:
            row.update({
                "first_response_status": source.get("status"),
                "validation_errors": source.get("validation_errors", []),
            })
            rejected_rows.append(row)

    accepted_rows.sort(key=lambda row: row["expansion_order"])
    rejected_rows.sort(key=lambda row: row["expansion_order"])
    accepted = {
        "manifest": "RevealNav RxR first-response branch proposals",
        "revision": "rxr-multiview-first-response-accepted/1",
        "status": "PASS_WITH_FAIL_CLOSED_REJECTIONS",
        "source": {"path": str(INPUT.relative_to(ROOT)), "sha256": input_sha},
        "input_event_count": len(manifest["events"]),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "events": accepted_rows,
        "rejected_events": rejected_rows,
        "selection_policy": "first provider response only; text-length repair only",
        "provider_retries_used_for_selection": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(ACCEPTED, accepted)

    events = {row["event_id"]: row for row in manifest["events"]}
    prescreen_rows = []
    rejected_ids = {row["event_id"] for row in rejected_rows}
    rejected_by_id = {row["event_id"]: row for row in rejected_rows}
    accepted_by_id = {row["event_id"]: row for row in accepted_rows}
    for event_id, event in sorted(events.items(), key=lambda item: item[1]["expansion_order"]):
        if event_id in rejected_ids:
            rejected = rejected_by_id[event_id]
            prescreen_rows.append({
                "event_id": event_id,
                "expansion_order": event["expansion_order"],
                "episode_id": event["episode_id"],
                "scene_id": event["scene_id"],
                "candidate_interval": event["candidate_interval"],
                "proposal_path": rejected["proposal_path"],
                "proposal_sha256": rejected["proposal_sha256"],
                "prescreen_disposition": "AUTO_REJECT_INVALID_FIRST_RESPONSE",
                "prescreen_reasons": rejected["validation_errors"] or [
                    rejected["first_response_status"]],
                "geometry_verified": False,
                "causal_prefix_verified": False,
                "human_label": None,
                "training_label": False,
            })
            continue
        accepted_row = accepted_by_id[event_id]
        proposal = accepted_payloads[event_id]["normalized_proposal"]
        branches = proposal["branches"]
        target = proposal["target_resolution"]
        target_id = target["target_branch_id"]
        target_branch = next((row for row in branches
                              if row["branch_id"] == target_id), None)
        alternatives = [
            row["branch_id"] for row in branches
            if row["branch_id"] != target_id
            and row["traversability_from_images"] == "LIKELY_TRAVERSABLE"
        ]
        flags = sorted(key for key in HARD_FLAGS if proposal["flags"][key])
        reasons = []
        if proposal["decision_status"] != "DECISION": reasons.append("MLLM_NOT_DECISION")
        if target["status"] != "UNIQUE" or target_branch is None: reasons.append("TARGET_NOT_UNIQUE")
        if len(branches) < 2: reasons.append("FEWER_THAN_TWO_VISUAL_BRANCHES")
        if target_branch is not None and target_branch["traversability_from_images"] != "LIKELY_TRAVERSABLE": reasons.append("TARGET_NOT_LIKELY_TRAVERSABLE_FROM_IMAGES")
        if not alternatives: reasons.append("NO_LIKELY_TRAVERSABLE_VISUAL_ALTERNATIVE")
        reasons.extend("HARD_FLAG_" + key.upper() for key in flags)
        disposition = ("AUTO_REJECT_BEFORE_3D" if reasons else
                       "RELOCATE_EARLIER_THEN_3D" if proposal["flags"]["already_visible_before_seed"] else
                       "TO_DIRECTED_GEOMETRY")
        prescreen_rows.append({
            "event_id": event_id,
            "expansion_order": event["expansion_order"],
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "candidate_interval": event["candidate_interval"],
            "proposal_path": accepted_row["proposal_path"],
            "proposal_sha256": accepted_row["proposal_sha256"],
            "mllm_decision_status": proposal["decision_status"],
            "mllm_target_status": target["status"],
            "mllm_target_branch_id": target_id,
            "mllm_branch_count": len(branches),
            "mllm_likely_alternative_branch_ids": alternatives,
            "mllm_already_visible_before_seed": proposal["flags"]["already_visible_before_seed"],
            "mllm_hard_flags": flags,
            "prescreen_disposition": disposition,
            "prescreen_reasons": reasons,
            "geometry_verified": False,
            "causal_prefix_verified": False,
            "human_label": None,
            "training_label": False,
        })
    counts = Counter(row["prescreen_disposition"] for row in prescreen_rows)
    prescreen = {
        "manifest": "RevealNav RxR first-response machine branch prescreen",
        "revision": "rxr-multiview-first-response-prescreen/1",
        "status": "PRESCREEN_COMPLETE_GEOMETRY_REQUIRED",
        "sources": {
            "multiview_input": {"path": str(INPUT.relative_to(ROOT)), "sha256": input_sha},
            "accepted_proposals": {"path": str(ACCEPTED.relative_to(ROOT)), "sha256": sha256_file(ACCEPTED)},
        },
        "event_count": len(prescreen_rows),
        "disposition_counts": dict(sorted(counts.items())),
        "events": prescreen_rows,
        "provider_retries_used_for_selection": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(PRESCREEN, prescreen)
    print(json.dumps({
        "status": prescreen["status"],
        "accepted_first_responses": len(accepted_rows),
        "rejected_first_responses": len(rejected_rows),
        "disposition_counts": prescreen["disposition_counts"],
        "prescreen_sha256": sha256_file(PRESCREEN),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
