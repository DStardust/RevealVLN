#!/usr/bin/env python3
"""Prepare isolated Human/Qwen/Codex forms for the MF2-CR3 review gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
PACKET = ROOT / ("artifacts/phase0/phase0c_language_review_35_v2_localmap/"
                 "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json")
CR3 = ROOT / "METHOD_FREEZE_2_CORRECTNESS_REVISION_3.md"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_hybrid_review"
EXPECTED_PACKET_SHA = \
    "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c"
TRACKS = {
    "H": {
        "filename": "TRACK_H_HUMAN.csv", "display_alias": "Reviewer A",
        "reviewer_type": "human", "reviewer_system": "project researcher",
    },
    "M1": {
        "filename": "TRACK_M1_QWEN38MAX.csv", "display_alias": "Reviewer B",
        "reviewer_type": "model",
        "reviewer_system": "Qwen3.8-Max under Claude-Code control",
    },
    "M2": {
        "filename": "TRACK_M2_CODEX.csv", "display_alias": "Reviewer C",
        "reviewer_type": "model",
        "reviewer_system": "clean Codex session",
    },
}

CORE_FIELDS = [
    "row_order", "event_id", "episode_id", "scene_id", "instruction_id",
    "instruction_sha256", "language", "instruction_text_for_private_review",
    "screening_triggers",
    "semantic_branch_id", "target_exit_region", "causal_prefixes",
    "private_media", "private_local_map", "private_review_board",
    "private_contact_sheet", "local_map_geometry", "instruction_render",
    "frozen_cost_frontiers",
    "immutable_row_sha256",
]
PROVENANCE_FIELDS = [
    "reviewer_track", "display_alias", "reviewer_type", "reviewer_system",
    "reviewer_version", "prompt_sha256", "packet_sha256",
]
JUDGMENT_FIELDS = [
    "reviewed", "reviewer_id", "review_timestamp",
    "branch_dependent_instruction", "instruction_clause",
    "target_branch_matches_instruction", "causal_reveal_confirmed",
    "semantic_track_confirmed", "cost_expiry_interpretation_confirmed",
    "candidate_valid", "rejection_reason", "reviewer_notes",
]
ALL_FIELDS = CORE_FIELDS + PROVENANCE_FIELDS + JUDGMENT_FIELDS

PROMPT = """# MF2-CR3 fixed independent review prompt

You are one isolated reviewer of a fixed 35-row private RxR-train engineering
packet. Review every row independently. Do not access or infer another track's
answers, do not change event metadata or thresholds, and do not use any
val-unseen/test data. Matterport-derived media and instructions are private.

For each row, read the private instruction and inspect the four ordered causal
front views (`P: pre-reveal`, `D1`, `D2`, `D3`) in the v2 review board. The
lower local map shows navigability, physical headings/63-degree view wedges,
the C1--C3 automatic candidate endpoints and the fixed green B-to-TARGET
semantic exit. P is the pre-reveal camera pose; B is the distinct beginning
of the fixed post-turn branch segment and need not coincide with P. Light
gray is the event-height navmesh; blue-gray, when shown, is an auxiliary
branch height for stairs. The map is an offline identity aid
only. It must never be used as evidence that the evaluated agent causally saw
the branch.

Set these six fields strictly to `true` or `false`:

1. `branch_dependent_instruction`: completing the instruction genuinely
   depends on taking the proposed directed branch rather than merely passing a
   generic nearby region.
2. `target_branch_matches_instruction`: the offline proposed exit region is
   the branch demanded by the instruction.
3. `causal_reveal_confirmed`: the target branch is unavailable/unclear in the
   pre-reveal frame and becomes visually actionable in the ordered D1--D3
   causal frames, without relying on the offline panel.
4. `semantic_track_confirmed`: the proposal set represents one directed
   executable exit, not two semantic alternatives.
5. `cost_expiry_interpretation_confirmed`: the supplied budget frontiers form
   a meaningful resource-conditioned last-passage/option-loss example; do not
   interpret it as an online-observable first-passage time.
6. `candidate_valid`: true only if every criterion above is true and you are
   confident. Any uncertainty must be false.

Fill `instruction_clause` with the minimal clause supporting your judgment.
For an invalid or uncertain row, fill `rejection_reason`; for a valid row,
leave it empty. `reviewer_notes` is optional. Set `reviewed=true`, supply your
assigned reviewer ID, exact system/model version, and an ISO-8601 timestamp.
Do not alter any core or prefilled provenance field. Do not tune this rubric
after viewing aggregate results.

Your controller will provide exactly one assigned track CSV. Write judgments
only to that file and retain a raw execution log. Do not open either other
track CSV or any human/model seal/result file.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def immutable_payload(row):
    keys = [
        "row_order", "event_id", "episode_id", "scene_id", "instruction_id",
        "instruction_sha256", "language", "screening_triggers",
        "semantic_branch_id", "target_exit_region", "causal_prefixes",
        "private_media", "private_local_map", "private_review_board",
        "private_contact_sheet", "local_map_geometry", "instruction_render",
        "frozen_cost_frontiers",
    ]
    return {key: row[key] for key in keys}


def main() -> int:
    if sha256_file(PACKET) != EXPECTED_PACKET_SHA:
        raise SystemExit("fixed packet SHA drift")
    packet = json.loads(PACKET.read_text())
    rows = packet.get("rows", [])
    if (packet.get("status") != "PASS_PENDING_HUMAN_REVIEW"
            or len(rows) != 35 or packet.get("reviewed_true_count") != 0):
        raise SystemExit("unexpected packet state")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = OUT_DIR / "FIXED_REVIEW_PROMPT.md"
    prompt_path.write_text(PROMPT)
    prompt_sha = sha256_file(prompt_path)

    form_records = []
    immutable_rows = {}
    for track, metadata in TRACKS.items():
        path = OUT_DIR / metadata["filename"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALL_FIELDS)
            writer.writeheader()
            for source in rows:
                immutable = immutable_payload(source)
                immutable_sha = sha256_bytes(
                    stable_json(immutable).encode("utf-8"))
                immutable_rows[source["event_id"]] = immutable_sha
                output = {
                    "row_order": source["row_order"],
                    "event_id": source["event_id"],
                    "episode_id": source["episode_id"],
                    "scene_id": source["scene_id"],
                    "instruction_id": source["instruction_id"],
                    "instruction_sha256": source["instruction_sha256"],
                    "language": source["language"],
                    "instruction_text_for_private_review":
                        source["instruction_text_for_private_review"],
                    "screening_triggers": stable_json(
                        source["screening_triggers"]),
                    "semantic_branch_id": source["semantic_branch_id"],
                    "target_exit_region": stable_json(
                        source["target_exit_region"]),
                    "causal_prefixes": stable_json(source["causal_prefixes"]),
                    "private_media": stable_json(source["private_media"]),
                    "private_local_map": source["private_local_map"],
                    "private_review_board": source["private_review_board"],
                    "private_contact_sheet": source["private_contact_sheet"],
                    "local_map_geometry": stable_json(
                        source["local_map_geometry"]),
                    "instruction_render": stable_json(
                        source["instruction_render"]),
                    "frozen_cost_frontiers": stable_json(
                        source["frozen_cost_frontiers"]),
                    "immutable_row_sha256": immutable_sha,
                    "reviewer_track": track,
                    "display_alias": metadata["display_alias"],
                    "reviewer_type": metadata["reviewer_type"],
                    "reviewer_system": metadata["reviewer_system"],
                    "reviewer_version": "",
                    "prompt_sha256": prompt_sha,
                    "packet_sha256": EXPECTED_PACKET_SHA,
                }
                output.update({field: "" for field in JUDGMENT_FIELDS})
                writer.writerow(output)
        form_records.append({
            "track": track, "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size, "blank_sha256": sha256_file(path),
            **metadata,
        })

    readme = OUT_DIR / "EXECUTION_README.md"
    readme.write_text("""# MF2-CR3 混合审核执行流程

1. 人类审核者先独立完成 `TRACK_H_HUMAN.csv`，期间不得打开任何模型输出；
   完成后交由主 agent 立即计算 SHA-256 封存。
2. Qwen 和全新的 Codex 会话使用同一份英文
   `FIXED_REVIEW_PROMPT.md`，并且各自只能写入被分配的 CSV。两者都使用
   中文图例审核板，但不得读取 H 或另一个模型的结果。
3. 三个轨道完成前，禁止 H 查看 M1/M2、M1 查看 H/M2、M2 查看 H/M1。
4. 三份结果由最终验证器做一致性汇总；只有三方六项判断全部为真的事件
   才接纳，分歧项保留并排除，禁止人工覆盖成通过。

A/B/C 仅为展示别名，机器来源中继续如实保留 human/model 类型。
""")
    human_guide = OUT_DIR / "HUMAN_REVIEW_GUIDE_ZH.md"
    human_guide.write_text("""# Human 轨道中文审核指南

## 审核顺序

只打开 `TRACK_H_HUMAN.csv` 和其中 `private_review_board` 指向的新版审核
板。不要查看 Qwen/Codex 的表或输出。35 项必须全部审核。

审核板左侧上方是 P（观察前）与 D1–D3（三次有序因果观察）；左侧
下方是局部地图和中文图例；整个最右侧逐字显示完整私有导航指令。判断
“是否被因果看到”时只能使用 P/D1–D3，地图只用于确认 C1–C3 是否对应
绿色 B→T。`P` 是观察前相机位置，`B` 是目标分支入口，两者不要求重合。

## CSV 填写要求

- `reviewed`：完成该行后填 `true`。
- `reviewer_id`：填写你的匿名审核者 ID，例如 `human_reviewer_A`。
- `review_timestamp`：填写带时区的 ISO-8601 时间。
- `reviewer_version`：Human 轨道填写 `human-v1`。
- `instruction_clause`：复制支持判定的最短指令片段。
- `rejection_reason`：无效或不确定时必填；有效时留空。
- `reviewer_notes`：可选。

六个布尔字段：

1. `branch_dependent_instruction`：指令确实要求走该分支。
2. `target_branch_matches_instruction`：绿色 B→T 与指令要求一致。
3. `causal_reveal_confirmed`：P 中不明确，D1–D3 中才变得可行动。
4. `semantic_track_confirmed`：C1–C3 表示同一个语义出口。
5. `cost_expiry_interpretation_confirmed`：延迟导致返回成本/预算风险的
   语义合理；不需要重算机器成本。
6. `candidate_valid`：只有前五项全部为真且你有把握时才填 `true`。

任何不确定均按 `false` 处理。不得修改事件、场景、指令、媒体路径、提示词
哈希、审核包哈希或 reviewer type/system 等预填字段。
""")
    manifest = {
        "manifest": "MF2-CR3 fixed hybrid-review package",
        "revision": "phase0c-hybrid-review-package/2-wide-instruction",
        "status": "READY_UNREVIEWED",
        "protocol": {
            "path": str(CR3.relative_to(ROOT)),
            "sha256": sha256_file(CR3),
        },
        "packet": {
            "path": str(PACKET.relative_to(ROOT)),
            "sha256": EXPECTED_PACKET_SHA,
            "rows": 35,
            "scenes": len({row["scene_id"] for row in rows}),
        },
        "prompt": {
            "path": str(prompt_path.relative_to(ROOT)),
            "sha256": prompt_sha,
        },
        "forms": form_records,
        "human_guide": {
            "path": str(human_guide.relative_to(ROOT)),
            "sha256": sha256_file(human_guide),
            "language": "zh-CN",
        },
        "llm_prompt_language": "en",
        "immutable_row_sha256": immutable_rows,
        "fixed_decision": {
            "rule": "unanimous six-boolean TRUE across H, M1 and M2",
            "all_rows_must_be_reviewed": 35,
            "minimum_accepted_events": 15,
            "minimum_accepted_scenes": 10,
            "disagreements": "retained and excluded",
        },
        "human_judgments_present": 0,
        "model_judgments_present": 0,
        "training_authorized": False,
        "private_distribution_authorized": False,
    }
    manifest_path = OUT_DIR / "HYBRID_REVIEW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"], "rows": 35,
        "forms": [item["path"] for item in form_records],
        "prompt_sha256": prompt_sha,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
