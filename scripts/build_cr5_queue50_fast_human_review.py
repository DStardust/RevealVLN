#!/usr/bin/env python3
"""Build a compact, unlabeled Chinese review packet for queue50 survivors."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_phase0c_cr5_human_review as board  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/multiview_primary"
INPUT = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
ACCEPTED = BASE / "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json"
PRESCREEN = BASE / "CR5_QUEUE50_PRIMARY_MACHINE_PRESCREEN.json"
GEOMETRY = BASE / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
CONTROLLER = BASE / "CR5_QUEUE50_CONTROLLER_EXECUTION.json"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/human_review_fast"
BOARD_DIR = OUT_DIR / "boards"
MANIFEST = OUT_DIR / "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json"
TEMPLATE = OUT_DIR / "CR5_QUEUE50_FAST_REVIEW_TEMPLATE.jsonl"
REJECTED = OUT_DIR / "CR5_QUEUE50_AUTO_REJECTED.json"
GUIDE = OUT_DIR / "审核说明.md"

# The instruction remains on the right, but the canvas is wider than the old
# pilot so even long RxR instructions fit without truncation.
board.CANVAS = (3600, 1800)
board.RIGHT_X = 2050


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    inputs = load(INPUT)
    accepted = load(ACCEPTED)
    prescreen = load(PRESCREEN)
    geometry = load(GEOMETRY)
    controller = load(CONTROLLER)
    event_by_id = {row["event_id"]: row for row in inputs["events"]}
    accepted_by_id = {row["event_id"]: row for row in accepted["events"]}
    geometry_by_id = {row["event_id"]: row for row in geometry["events"]}
    controller_by_id = {row["event_id"]: row
                        for row in controller["events"]}
    if len(event_by_id) != 50 or set(event_by_id) != set(accepted_by_id):
        raise SystemExit("queue50 input closure failure")
    survivors = [row for row in controller["events"] if row["status"] ==
                 "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"]
    if len(survivors) != 34:
        raise SystemExit("expected 34 controller-pass review candidates")

    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    template_rows = []
    for result in sorted(survivors,
                         key=lambda row: event_by_id[row["event_id"]][
                             "queue_order"]):
        event_id = result["event_id"]
        event = event_by_id[event_id]
        geometry_row = geometry_by_id[event_id]
        accepted_row = accepted_by_id[event_id]
        proposal_path = ROOT / accepted_row["accepted_proposal_path"]
        if sha256_file(proposal_path) != accepted_row[
                "accepted_proposal_sha256"]:
            raise SystemExit("proposal SHA drift: " + event_id)
        proposal = load(proposal_path)["normalized_proposal"]
        angle = geometry_row["alternative"]["distinctness"][
            "three_dimensional_angle_at_1m_deg"]
        separation = geometry_row["alternative"]["distinctness"][
            "separation_at_1_75m_m"]
        confidence = float(proposal["confidence"])
        strong = angle >= 60.0 and separation >= 1.8
        tier = "GEOMETRY_STRONG" if strong else "GEOMETRY_BORDERLINE"
        note = (
            "已通过自动分支语义、3D 导航网格和离散控制器双重回放；"
            "请只核对图像语义、指令唯一性与 Q 时刻是否合理。"
        )
        image = board.build_board(event, proposal, geometry_row, result,
                                  tier, note)
        path = BOARD_DIR / (event_id + "_review.jpg")
        image.save(path, format="JPEG", quality=92, subsampling=0,
                   optimize=True)
        rows.append({
            "queue_order": event["queue_order"],
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "instruction_text": event["instruction_text"],
            "instruction_sha256": event["instruction_sha256"],
            "proposal_path": accepted_row["accepted_proposal_path"],
            "proposal_sha256": accepted_row[
                "accepted_proposal_sha256"],
            "board_path": str(path.relative_to(ROOT)),
            "board_bytes": path.stat().st_size,
            "board_sha256": sha256_file(path),
            "board_pixels": list(board.CANVAS),
            "automatic_review_priority": tier,
            "mllm_confidence_diagnostic": confidence,
            "branch_angle_deg": angle,
            "branch_separation_at_1_75m_m": separation,
            "target_branch_id": geometry_row["target"]["branch_id"],
            "alternative_branch_id": geometry_row["alternative"][
                "branch_id"],
            "geometry_status": geometry_row["status"],
            "controller_status": result["status"],
            "causal_prefix_status": "PENDING_SEPARATE_GATE",
            "human_status": "PENDING",
            "training_label": False,
        })
        template_rows.append({
            "reviewer_id": None,
            "reviewer_type": "HUMAN",
            "event_id": event_id,
            "two_distinct_executable_exits": None,
            "alternative_is_not_incoming_closed_or_duplicate": None,
            "instruction_uniquely_selects_target": None,
            "decision_center_and_temporal_order_are_reasonable": None,
            "final_label": None,
            "reason_codes": [],
            "comment_zh": "",
        })

    rejected_rows = []
    for event in sorted(inputs["events"], key=lambda row: row["queue_order"]):
        event_id = event["event_id"]
        if event_id in {row["event_id"] for row in survivors}:
            continue
        geometry_row = geometry_by_id[event_id]
        if geometry_row["status"] == "GEOMETRY_REJECT":
            stage = "DIRECTED_GEOMETRY"
            reasons = geometry_row["failures"]
        else:
            stage = "DISCRETE_CONTROLLER"
            controller_row = controller_by_id[event_id]
            reasons = []
            for role in ("target", "alternative"):
                branch = controller_row[role]
                if not branch["pass"]:
                    replay_statuses = sorted({row["status"]
                                              for row in branch["replays"]})
                    reasons.append(role.upper() + "_" +
                                   "_".join(replay_statuses))
            if not reasons:
                reasons = ["CONTROLLER_NONDETERMINISTIC_OR_FAILED"]
        rejected_rows.append({
            "queue_order": event["queue_order"],
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "automatic_reject_stage": stage,
            "automatic_reject_reasons": reasons,
            "human_spot_check_status": "OPTIONAL_PENDING",
            "human_label": None,
            "training_label": False,
        })
    reject_counts = Counter(row["automatic_reject_stage"]
                            for row in rejected_rows)
    reject_output = {
        "manifest": "MF2-CR5 queue50 automatic rejection ledger",
        "status": "MACHINE_REJECTIONS_NOT_HUMAN_LABELS",
        "rejected_count": len(rejected_rows),
        "stage_counts": dict(sorted(reject_counts.items())),
        "recommended_human_spot_check_count": 4,
        "events": rejected_rows,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(REJECTED, reject_output)

    manifest = {
        "manifest": "MF2-CR5 queue50 fast human branch review",
        "revision": "cr5-queue50-fast-human-review/1",
        "status": "READY_FOR_HUMAN_BRANCH_REVIEW",
        "screened_trajectory_count": 50,
        "full_review_board_count": len(rows),
        "automatic_reject_count": len(rejected_rows),
        "recommended_reject_spot_check_count": 4,
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (INPUT, ACCEPTED, PRESCREEN, GEOMETRY, CONTROLLER,
                         board.FONT, board.FONT_LICENSE)
        },
        "items": rows,
        "automatic_rejection_ledger": {
            "path": str(REJECTED.relative_to(ROOT)),
            "sha256": sha256_file(REJECTED),
        },
        "labels_created": 0,
        "human_reviews_completed": 0,
        "frozen_50_item_human_protocol_satisfied": False,
        "causal_prefix_gate_completed": False,
        "training_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    TEMPLATE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in template_rows))
    GUIDE.write_text(
        "# 50 条扩展的快速人工审核说明\n\n"
        "本轮已自动筛查 50 条冻结轨迹。12 条在 3D 几何关卡失败，4 条在真实离散控制器回放失败；"
        "这些机器拒绝不是人工标签。你只需完整审核 `boards/` 中的 34 张图，并建议从 16 条拒绝中抽查 4 条。\n\n"
        "对每张图只回答四个问题：\n\n"
        "1. 绿色目标和紫色备选是否是两条真正不同、能够通行的出口？\n"
        "2. 紫色备选是否不是来路、关闭门、同一出口的另一视角或很短的假分叉？\n"
        "3. 右侧完整指令是否能唯一选中绿色目标？\n"
        "4. A→Q→D 是否说明 Q 正好位于需要作选择的位置？\n\n"
        "四项都为是，填 `ACCEPT`；任一项明确为否，填 `REJECT`；图像证据不足，填 `AMBIGUOUS`。"
        "模板为 `CR5_QUEUE50_FAST_REVIEW_TEMPLATE.jsonl`，所有字段当前均为空。\n",
        encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "screened": 50,
        "full_review_boards": len(rows),
        "automatic_rejects": len(rejected_rows),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST),
        "template_sha256": sha256_file(TEMPLATE),
        "guide_sha256": sha256_file(GUIDE),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
