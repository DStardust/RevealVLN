#!/usr/bin/env python3
"""Seal valid expansion branch proposals and machine-prescreen them."""

from __future__ import annotations

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
RUN_DIR = BASE / "branch_factory/runs"
ACCEPTED = BASE / "branch_factory/RXR_MULTIVIEW_BRANCH_ACCEPTED.json"
PRESCREEN = BASE / "branch_factory/RXR_MULTIVIEW_MACHINE_PRESCREEN.json"
SHARDS = 28
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
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if not INPUT.is_file() or INPUT.is_symlink():
        raise SystemExit("multiview input is not ready")
    input_sha = sha256_file(INPUT)
    manifest = json.loads(INPUT.read_text())
    if (manifest.get("status") != "READY_FOR_BRANCH_PROPOSER"
            or manifest.get("branch_labels_created") != 0
            or manifest.get("training_authorized") is not False):
        raise SystemExit("multiview input contract failure")
    events = {row["event_id"]: row for row in manifest["events"]}
    accepted_rows = []
    run_sources = []
    observed_ids = set()
    for index in range(SHARDS):
        run_path = RUN_DIR / ("shard_%02d.json" % index)
        if not run_path.is_file() or run_path.is_symlink():
            raise SystemExit("missing branch shard: " + str(index))
        run = json.loads(run_path.read_text())
        expected_ids = {row["event_id"] for row in events.values()
                        if row["expansion_order"] % SHARDS == index}
        shard_ids = {row["event_id"] for row in run.get("results", [])}
        if (run.get("status") != "PASS"
                or run.get("input_sha256") != input_sha
                or run.get("shard_index") != index
                or run.get("shard_count") != SHARDS
                or run.get("enable_thinking") is not False
                or run.get("reasoning_effort") != "none"
                or shard_ids != expected_ids):
            raise SystemExit("branch shard contract failure: " + str(index))
        for row in run["results"]:
            if row["event_id"] in observed_ids:
                raise SystemExit("duplicate accepted event")
            event = events[row["event_id"]]
            path = ROOT / row["path"]
            if (row.get("status") != "VALID_MLLM_PROPOSAL"
                    or not path.is_file() or path.is_symlink()
                    or sha256_file(path) != row["sha256"]):
                raise SystemExit("invalid accepted branch result")
            value = json.loads(path.read_text())
            adapted = factory.contract_event(event)
            if (value.get("status") != "VALID_MLLM_PROPOSAL"
                    or value.get("provider_model") != factory.MODEL
                    or value.get("enable_thinking") is not False
                    or value["request_evidence"].get("input_sha256") !=
                    input_sha
                    or contract.validate_proposal(
                        value["normalized_proposal"], adapted)):
                raise SystemExit("accepted payload contract failure")
            for media in value["request_evidence"]["media"]:
                factory.safe_media(media)
            observed_ids.add(row["event_id"])
            accepted_rows.append({
                "event_id": row["event_id"],
                "expansion_order": event["expansion_order"],
                "episode_id": event["episode_id"],
                "accepted_proposal_path": row["path"],
                "accepted_proposal_sha256": row["sha256"],
                "human_reviewed": False,
                "training_label": False,
            })
        run_sources.append({
            "path": str(run_path.relative_to(ROOT)),
            "sha256": sha256_file(run_path),
            "job_count": run["job_count"],
        })
    if observed_ids != set(events):
        raise SystemExit("branch accepted exact closure failure")
    accepted_rows.sort(key=lambda row: row["expansion_order"])
    accepted = {
        "manifest": "RevealNav RxR expansion accepted branch proposals",
        "revision": "rxr-multiview-branch-accepted/1",
        "status": "PASS",
        "sources": {
            "multiview_input": {"path": str(INPUT.relative_to(ROOT)),
                                "sha256": input_sha},
            "shards": run_sources,
        },
        "event_count": len(accepted_rows),
        "accepted_count": len(accepted_rows),
        "events": accepted_rows,
        "offline_annotation_only": True,
        "geometry_labels_created": 0,
        "online_causal_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(ACCEPTED, accepted)

    prescreen_rows = []
    for accepted_row in accepted_rows:
        event = events[accepted_row["event_id"]]
        payload = json.loads(
            (ROOT / accepted_row["accepted_proposal_path"]).read_text())
        proposal = payload["normalized_proposal"]
        branches = proposal["branches"]
        target = proposal["target_resolution"]
        target_id = target["target_branch_id"]
        target_branch = next((row for row in branches
                              if row["branch_id"] == target_id), None)
        likely_alternatives = [
            row["branch_id"] for row in branches
            if row["branch_id"] != target_id
            and row["traversability_from_images"] == "LIKELY_TRAVERSABLE"]
        active_hard_flags = sorted(
            key for key in HARD_FLAGS if proposal["flags"][key])
        reasons = []
        if proposal["decision_status"] != "DECISION":
            reasons.append("MLLM_NOT_DECISION")
        if target["status"] != "UNIQUE" or target_branch is None:
            reasons.append("TARGET_NOT_UNIQUE")
        if len(branches) < 2:
            reasons.append("FEWER_THAN_TWO_VISUAL_BRANCHES")
        if (target_branch is not None and target_branch[
                "traversability_from_images"] != "LIKELY_TRAVERSABLE"):
            reasons.append("TARGET_NOT_LIKELY_TRAVERSABLE_FROM_IMAGES")
        if not likely_alternatives:
            reasons.append("NO_LIKELY_TRAVERSABLE_VISUAL_ALTERNATIVE")
        reasons.extend("HARD_FLAG_" + key.upper()
                       for key in active_hard_flags)
        if reasons:
            disposition = "AUTO_REJECT_BEFORE_3D"
        elif proposal["flags"]["already_visible_before_seed"]:
            disposition = "RELOCATE_EARLIER_THEN_3D"
        else:
            disposition = "TO_DIRECTED_GEOMETRY"
        prescreen_rows.append({
            "event_id": event["event_id"],
            "expansion_order": event["expansion_order"],
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "candidate_interval": event["candidate_interval"],
            "proposal_path": accepted_row["accepted_proposal_path"],
            "proposal_sha256": accepted_row["accepted_proposal_sha256"],
            "mllm_decision_status": proposal["decision_status"],
            "mllm_target_status": target["status"],
            "mllm_target_branch_id": target_id,
            "mllm_branch_count": len(branches),
            "mllm_likely_alternative_branch_ids": likely_alternatives,
            "mllm_already_visible_before_seed": proposal["flags"][
                "already_visible_before_seed"],
            "mllm_hard_flags": active_hard_flags,
            "prescreen_disposition": disposition,
            "prescreen_reasons": reasons,
            "prescreen_is_machine_only": True,
            "geometry_verified": False,
            "causal_prefix_verified": False,
            "human_label": None,
            "training_label": False,
        })
    counts = Counter(row["prescreen_disposition"]
                     for row in prescreen_rows)
    prescreen = {
        "manifest": "RevealNav RxR expansion machine branch prescreen",
        "revision": "rxr-multiview-machine-prescreen/1",
        "status": "PRESCREEN_COMPLETE_GEOMETRY_REQUIRED",
        "sources": {
            "multiview_input": {"path": str(INPUT.relative_to(ROOT)),
                                "sha256": input_sha},
            "accepted_proposals": {
                "path": str(ACCEPTED.relative_to(ROOT)),
                "sha256": sha256_file(ACCEPTED),
            },
        },
        "policy": {
            "hard_flags": sorted(HARD_FLAGS),
            "already_visible_action": "relocate earlier before causal gate",
            "requires_unique_target": True,
            "requires_likely_traversable_target": True,
            "requires_likely_traversable_alternative": True,
            "mllm_is_offline_proposal_only": True,
            "machine_prescreen_is_not_human_truth": True,
            "geometry_and_causal_controller_still_required": True,
        },
        "event_count": len(prescreen_rows),
        "disposition_counts": dict(sorted(counts.items())),
        "events": prescreen_rows,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(PRESCREEN, prescreen)
    print(json.dumps({
        "status": prescreen["status"],
        "accepted": len(accepted_rows),
        "disposition_counts": prescreen["disposition_counts"],
        "accepted_sha256": sha256_file(ACCEPTED),
        "prescreen_sha256": sha256_file(PRESCREEN),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
