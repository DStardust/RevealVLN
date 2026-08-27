#!/usr/bin/env python3
"""Fail-closed prescreen for CR5 MLLM multi-view branch proposals.

This stage is deliberately weaker than geometry verification.  It only decides
which proposals are worth grounding, which must be relocated earlier on the
reference trace, and which are already contradicted by their own structured
output or by the recorded main-agent visual prescreen.  It never creates a
training label.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
INPUT = BASE / "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
RUN = BASE / "CR5_MULTIVIEW_PREFLIGHT_RUN_V2.json"
PROPOSALS = BASE / "proposals_v2"
OUT = BASE / "CR5_MULTIVIEW_MAIN_AGENT_PRESCREEN_V2.json"

EXPECTED_INPUT_SHA256 = (
    "3d3a1d4ce468c8a54a5a61b96f340a415bad8357442ae242b0cf6b595a12f7fe"
)
EXPECTED_RUN_SHA256 = (
    "b7cc5514e39a073e363062abbb5374e2ab59d0183d949c3f90bb306ae4d83823"
)

# V00 is reference-route-forward.  A target labelled behind the reference
# direction is inconsistent with the directed grounding contract.  BACK*
# branches are also excluded when counting non-incoming alternatives.
FORWARD_HEMISPHERE = {
    "FRONT", "FRONT_LEFT", "LEFT", "FRONT_RIGHT", "RIGHT",
}
HARD_FLAG_NAMES = {
    "no_alternative_exit", "retrace_only", "single_channel_turn",
    "target_language_ambiguous", "visual_evidence_ambiguous",
}

# These are explicit main-agent image prescreen judgements, not human ground
# truth and not reusable benchmark labels.  Each remains auditable against the
# immutable panorama SHA records in INPUT.
VISUAL_OVERRIDES = {
    "ep41233_hv03": {
        "decision": "REJECT",
        "reason": "STAIR_SWITCHBACK_OR_RETRACE_CONFLATION",
        "note_zh": "楼梯折返平台被枚举成多条分支；未见两个独立的前向出口。",
    },
    "ep43805_hv04": {
        "decision": "REJECT",
        "reason": "BLOCKED_DOORS_AND_RETRACE_ONLY",
        "note_zh": "画面中的额外门为关闭/阻断状态，其余方向主要是来路；没有第二条可执行前向出口。",
    },
    "ep34121_hv04": {
        "decision": "REJECT",
        "reason": "POST_BRANCH_SINGLE_FORWARD_ROOM",
        "note_zh": "已经越过上一个平台分支，当前只剩进入房间的前向通道和来路。",
    },
    "ep46758_hv03": {
        "decision": "REJECT",
        "reason": "POST_BRANCH_OPEN_ROOM_WITHOUT_DISCRETE_EXITS",
        "note_zh": "已经进入目标起居区；宽阔室内的多个朝向不等于离散出口。",
    },
    "ep56443_hv01": {
        "decision": "REJECT",
        "reason": "OPEN_ROOM_VIEWPOINT_NOT_DECISION_REGION",
        "note_zh": "沙发区内的观察点，候选方向仍在同一开放空间中。",
    },
    "ep56443_hv02": {
        "decision": "REJECT",
        "reason": "WINDOW_GOAL_AND_INCOMING_NOT_TWO_EXITS",
        "note_zh": "窗边是目标区域而非出口，另一方向主要是来路。",
    },
    "ep56443_hv04": {
        "decision": "REJECT",
        "reason": "OPEN_ROOM_VIEWPOINT_NOT_DECISION_REGION",
        "note_zh": "同一开放客餐厨空间内的中间观察点，没有离散分支入口。",
    },
    "ep56443_hv07": {
        "decision": "REJECT",
        "reason": "OPEN_ROOM_VIEWPOINT_NOT_DECISION_REGION",
        "note_zh": "厨房岛台旁的连续开放空间，视角方向不能直接当作分支。",
    },
    "ep56443_hv08": {
        "decision": "REJECT",
        "reason": "ENDPOINT_WITHOUT_DOWNSTREAM_BRANCH_LENGTH",
        "note_zh": "接近轨迹终点，无法形成满足 1.5--2.0m 的下游目标分支。",
    },
    "ep7619_hv02": {
        "decision": "REJECT",
        "reason": "CLOSED_DOOR_PSEUDO_BRANCHES",
        "note_zh": "厨房一侧被模型枚举的门均为关闭状态；可执行方向不足两个。",
    },
    "ep7619_hv03": {
        "decision": "REJECT",
        "reason": "SINGLE_CORRIDOR_WITH_CLOSED_SIDE_DOORS",
        "note_zh": "走廊两侧门关闭，只有前后单通道。",
    },
    "ep7619_hv07": {
        "decision": "REJECT",
        "reason": "LANDING_WITH_CLOSED_DOORS_AND_INCOMING_STAIR",
        "note_zh": "楼梯平台的额外门关闭，可执行方向退化为来路加单一出口。",
    },
    "ep7619_hv09": {
        "decision": "REJECT",
        "reason": "BEDROOM_ENDPOINT_WITHOUT_DECISION",
        "note_zh": "卧室终点观察，不是需要选择出口的决策区。",
    },
}

# Temporal candidates that describe one decision region.  The canonical member
# is chosen for the clearest decision-centred panorama; earlier members remain
# available to the later causal Reveal search and are not discarded.
SPATIAL_GROUPS = {
    "ep34121_bedroom_exit": {
        "members": ["ep34121_hv01", "ep34121_hv02"],
        "canonical": "ep34121_hv01",
    },
    "ep34121_balcony_room": {
        "members": ["ep34121_hv05", "ep34121_hv06"],
        "canonical": "ep34121_hv05",
    },
    "ep41233_stair_landing": {
        "members": ["ep41233_hv01", "ep41233_hv02"],
        "canonical": "ep41233_hv02",
    },
    "ep56443_dining_kitchen_arch": {
        "members": ["ep56443_hv05", "ep56443_hv06"],
        "canonical": "ep56443_hv05",
    },
    "ep7619_foyer_stair": {
        "members": ["ep7619_hv04", "ep7619_hv05", "ep7619_hv06"],
        "canonical": "ep7619_hv05",
    },
}

GROUP_BY_EVENT = {
    event_id: (group_id, group["canonical"])
    for group_id, group in SPATIAL_GROUPS.items()
    for event_id in group["members"]
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def stable_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA256:
        raise SystemExit("CR5 multi-view input SHA drift")
    if sha256_file(RUN) != EXPECTED_RUN_SHA256:
        raise SystemExit("CR5 multi-view run SHA drift")

    manifest = json.loads(INPUT.read_text())
    run = json.loads(RUN.read_text())
    if manifest.get("event_count") != 35:
        raise SystemExit("expected exactly 35 preflight events")
    if run.get("status") not in {"PASS", "COMPLETE", "VALID"}:
        # The historical summary uses a descriptive status.  Require its
        # explicit success counts below even if the spelling differs.
        if run.get("valid_count") != 35 and run.get("valid") != 35:
            raise SystemExit("MLLM run summary is not a 35/35 success")

    event_by_id = {row["event_id"]: row for row in manifest["events"]}
    rows = []
    proposal_shas = {}
    for event_id in sorted(event_by_id):
        path = PROPOSALS / (event_id + ".json")
        if not path.is_file() or path.is_symlink():
            raise SystemExit("missing/symlink proposal: " + event_id)
        proposal_shas[event_id] = sha256_file(path)
        record = json.loads(path.read_text())
        if (record.get("status") != "VALID_MLLM_PROPOSAL"
                or record.get("event_id") != event_id
                or record.get("validation_errors") != []
                or record.get("training_authorized") is not False):
            raise SystemExit("invalid proposal envelope: " + event_id)
        proposal = record["normalized_proposal"]
        branches = proposal["branches"]
        branch_by_id = {row["branch_id"]: row for row in branches}
        target = proposal["target_resolution"].get("target_branch_id")
        target_branch = branch_by_id.get(target)
        flags = proposal["flags"]
        likely_forward = [
            row for row in branches
            if row["traversability_from_images"] == "LIKELY_TRAVERSABLE"
            and row["horizontal_direction"] in FORWARD_HEMISPHERE
        ]
        hard_flags = sorted(name for name in HARD_FLAG_NAMES
                            if flags.get(name) is True)
        reasons = []
        if proposal["decision_status"] != "DECISION":
            reasons.append("MLLM_NOT_DECISION")
        if proposal["target_resolution"]["status"] != "UNIQUE":
            reasons.append("TARGET_NOT_UNIQUE")
        if target_branch is None:
            reasons.append("TARGET_BRANCH_MISSING")
        else:
            if (target_branch["traversability_from_images"] !=
                    "LIKELY_TRAVERSABLE"):
                reasons.append("TARGET_NOT_LIKELY_TRAVERSABLE")
            if target_branch["horizontal_direction"] not in FORWARD_HEMISPHERE:
                reasons.append("TARGET_OUTSIDE_ROUTE_FORWARD_HEMISPHERE")
        if len(likely_forward) < 2:
            reasons.append("FEWER_THAN_TWO_FORWARD_EXECUTABLE_PROPOSALS")
        reasons.extend("MLLM_FLAG_" + name.upper() for name in hard_flags)

        override = VISUAL_OVERRIDES.get(event_id)
        if override and override["decision"] == "REJECT":
            reasons.append("MAIN_AGENT_VISUAL_" + override["reason"])

        group = GROUP_BY_EVENT.get(event_id)
        if reasons:
            disposition = "HARD_REJECT_BEFORE_3D"
        elif group and event_id != group[1]:
            disposition = "SPATIAL_MERGE_MEMBER_NONCANONICAL"
        elif flags.get("already_visible_before_seed") is True:
            disposition = "RELOCATE_EARLIER_THEN_3D"
        else:
            disposition = "CAUSAL_CANDIDATE_TO_3D"

        rows.append({
            "event_id": event_id,
            "episode_id": event_by_id[event_id]["episode_id"],
            "scene_id": event_by_id[event_id]["scene_id"],
            "candidate_interval": event_by_id[event_id][
                "candidate_interval"],
            "proposal_sha256": proposal_shas[event_id],
            "mllm_decision_status": proposal["decision_status"],
            "mllm_target_status": proposal["target_resolution"]["status"],
            "mllm_target_branch_id": target,
            "mllm_branch_count": len(branches),
            "mllm_likely_forward_branch_ids": [
                row["branch_id"] for row in likely_forward],
            "mllm_already_visible_before_seed": flags.get(
                "already_visible_before_seed"),
            "mllm_hard_flags": hard_flags,
            "main_agent_visual_override": override,
            "spatial_group_id": group[0] if group else None,
            "spatial_group_canonical_event_id": group[1] if group else None,
            "prescreen_disposition": disposition,
            "prescreen_reasons": reasons,
            "geometry_verified": False,
            "causal_prefix_verified": False,
            "human_label": None,
            "training_label": False,
        })

    counts = Counter(row["prescreen_disposition"] for row in rows)
    output = {
        "manifest": "MF2-CR5 main-agent MLLM proposal prescreen",
        "revision": "cr5-main-agent-prescreen/2-all-panorama-audited",
        "status": "PRESCREEN_COMPLETE_GEOMETRY_REQUIRED",
        "scope": "six blinded RxR-train preflight trajectories only",
        "sources": {
            "multiview_input": {
                "path": str(INPUT.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256,
            },
            "mllm_run": {
                "path": str(RUN.relative_to(ROOT)),
                "sha256": EXPECTED_RUN_SHA256,
            },
            "proposal_directory": str(PROPOSALS.relative_to(ROOT)),
            "proposal_shas": proposal_shas,
        },
        "policy": {
            "forward_hemisphere": sorted(FORWARD_HEMISPHERE),
            "hard_flags": sorted(HARD_FLAG_NAMES),
            "already_visible_action": "relocate earlier; never accept at seed",
            "mllm_is_proposal_only": True,
            "visual_overrides_are_main_agent_prescreen_not_human_truth": True,
            "geometry_and_controller_execution_still_required": True,
            "spatial_groups": SPATIAL_GROUPS,
        },
        "event_count": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "events": rows,
        "geometry_verified_count": 0,
        "human_verified_count": 0,
        "training_authorized": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_suffix(".json.part")
    part.write_bytes(json.dumps(output, indent=2, ensure_ascii=False).encode(
        "utf-8") + b"\n")
    os.replace(part, OUT)
    print(json.dumps({
        "status": output["status"],
        "event_count": output["event_count"],
        "disposition_counts": output["disposition_counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
