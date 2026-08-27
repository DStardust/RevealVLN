#!/usr/bin/env python3
"""Fail-closed deterministic prescreen for queue50 branch proposals."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
INPUT = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
ACCEPTED = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json"
OUT = BASE / "CR5_QUEUE50_PRIMARY_MACHINE_PRESCREEN.json"
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


def load(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    manifest = load(INPUT)
    accepted = load(ACCEPTED)
    if accepted.get("status") != "PASS" or accepted.get("event_count") != 50:
        raise SystemExit("accepted proposal closure is not PASS/50")
    events = {row["event_id"]: row for row in manifest["events"]}
    accepted_rows = {row["event_id"]: row for row in accepted["events"]}
    if set(events) != set(accepted_rows) or len(events) != 50:
        raise SystemExit("manifest/accepted event closure mismatch")
    rows = []
    for event in manifest["events"]:
        event_id = event["event_id"]
        accepted_row = accepted_rows[event_id]
        proposal_path = ROOT / accepted_row["accepted_proposal_path"]
        if (not proposal_path.is_file() or proposal_path.is_symlink()
                or ROOT.resolve() not in proposal_path.resolve().parents):
            raise SystemExit("unsafe proposal: " + event_id)
        if sha256_file(proposal_path) != accepted_row[
                "accepted_proposal_sha256"]:
            raise SystemExit("proposal SHA drift: " + event_id)
        payload = load(proposal_path)
        if payload.get("status") != "VALID_MLLM_PROPOSAL":
            raise SystemExit("accepted proposal is not valid: " + event_id)
        proposal = payload["normalized_proposal"]
        branches = proposal["branches"]
        target = proposal["target_resolution"]
        target_id = target["target_branch_id"]
        target_branch = next(
            (row for row in branches if row["branch_id"] == target_id), None)
        alternatives = [row for row in branches
                        if row["branch_id"] != target_id]
        likely_alternatives = [row["branch_id"] for row in alternatives
                               if row["traversability_from_images"] ==
                               "LIKELY_TRAVERSABLE"]
        active_hard_flags = sorted(
            key for key in HARD_FLAGS if proposal["flags"][key])
        reasons = []
        if proposal["decision_status"] != "DECISION":
            reasons.append("MLLM_NOT_DECISION")
        if target["status"] != "UNIQUE" or target_branch is None:
            reasons.append("TARGET_NOT_UNIQUE")
        if len(branches) < 2:
            reasons.append("FEWER_THAN_TWO_VISUAL_BRANCHES")
        if target_branch is not None and target_branch[
                "traversability_from_images"] != "LIKELY_TRAVERSABLE":
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
        rows.append({
            "event_id": event_id,
            "queue_order": event["queue_order"],
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
    counts = Counter(row["prescreen_disposition"] for row in rows)
    output = {
        "manifest": "MF2-CR5 queue50 deterministic machine prescreen",
        "revision": "cr5-queue50-machine-prescreen/1",
        "status": "PRESCREEN_COMPLETE_GEOMETRY_REQUIRED",
        "sources": {
            "multiview_input": {
                "path": str(INPUT.relative_to(ROOT)),
                "sha256": sha256_file(INPUT),
            },
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
        "event_count": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "events": rows,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    part = OUT.with_suffix(".json.part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "counts": output["disposition_counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
