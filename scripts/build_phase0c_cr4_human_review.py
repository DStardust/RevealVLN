#!/usr/bin/env python3
"""Build the MF2-CR4 MLLM-assisted, human-verified private review packet."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/mnt/daiyang/vla")
LOCAL_PACKET = ROOT / (
    "artifacts/phase0/phase0c_language_review_35_v2_localmap/"
    "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json")
MLLM_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
MLLM_INPUT = MLLM_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS.json"
MLLM_ACCEPTANCE = MLLM_DIR / "MLLM_CLAUSE_GROUNDING_ACCEPTANCE.json"
CR4 = ROOT / "METHOD_FREEZE_2_CORRECTNESS_REVISION_4.md"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr4_human_review"
MEDIA_DIR = OUT_DIR / "private_media"
OUT = OUT_DIR / "PHASE0C_CR4_HUMAN_REVIEW.json"
CSV = OUT_DIR / "HUMAN_REVIEW_CR4.csv"
GUIDE = OUT_DIR / "HUMAN_REVIEW_GUIDE_CR4_ZH.md"
LLM_PROMPT = OUT_DIR / "FIXED_LLM_AUDIT_PROMPT_CR4.md"
FONT_PATH = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
EXPECTED = {
    LOCAL_PACKET:
        "3c3f650fa26ceb1d948614e3c1eb6800dca85504e1cad7690c52ab1294424c7c",
    MLLM_INPUT:
        "d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca",
    MLLM_ACCEPTANCE:
        "8a014c571b8d8715b057a547ff6c5ee409c358a70244ce0aa94919b485404bfb",
    CR4:
        "d052bcbf538586c1506ea1b9899f901342bccd369a256da3ea9173600a414a7f",
    FONT_PATH:
        "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b",
}
BOARD_WIDTH = 2560
TOP_HEIGHT = 920
TIMELINE_HEIGHT = 420
BOARD_HEIGHT = TOP_HEIGHT + TIMELINE_HEIGHT
EVIDENCE_WIDTH = 1120
INSTRUCTION_WIDTH = BOARD_WIDTH - EVIDENCE_WIDTH
TILE_SLOT_WIDTH = 150
TILE_IMAGE_SIZE = 132
TILES_PER_ROW = 17
HUMAN_FIELDS = [
    "reviewed", "reviewer_id", "review_timestamp",
    "clause_alignment_decision", "human_selected_segment_ids",
    "instruction_clause", "branch_dependent_instruction",
    "target_branch_matches_instruction", "causal_reveal_confirmed",
    "semantic_track_confirmed", "cost_expiry_interpretation_confirmed",
    "candidate_valid", "rejection_reason", "reviewer_notes",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size=size)


def wrap_text(draw, value: str, chosen_font, width: int):
    words = value.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textlength(candidate, font=chosen_font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def exact_wrap_ranges(draw, source: str, chosen_font, width: int):
    ranges = []
    start = 0
    while start < len(source):
        end = start
        last_space = None
        while end < len(source):
            end += 1
            if source[end - 1].isspace():
                last_space = end
            if draw.textlength(source[start:end], font=chosen_font) > width:
                if end - start == 1:
                    break
                end = last_space if last_space and last_space > start else end - 1
                break
        if end <= start:
            raise RuntimeError("instruction wrap made no progress")
        ranges.append((start, end))
        start = end
    if "".join(source[a:b] for a, b in ranges) != source:
        raise RuntimeError("instruction wrap was not lossless")
    return ranges


def instruction_panel(event, proposal):
    image = Image.new("RGB", (INSTRUCTION_WIDTH, TOP_HEIGHT), (249, 249, 249))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, INSTRUCTION_WIDTH - 1, 52), fill=(18, 18, 18))
    draw.text((16, 8), "CR4：完整指令 + MLLM 子句建议（仅建议）",
              font=font(26), fill=(255, 255, 255))
    selected = proposal["selected_segment_ids"]
    summary = ("状态：%s    建议子句：%s    置信度：%.2f" %
               (proposal["status"], ", ".join(selected) or "无",
                proposal["confidence"]))
    draw.text((18, 65), summary, font=font(22), fill=(30, 30, 30))
    draw.rectangle((16, 102, INSTRUCTION_WIDTH - 18, 106),
                   fill=(230, 190, 30))
    rationale_lines = wrap_text(draw, "模型理由：" + proposal["rationale"],
                                font(16), INSTRUCTION_WIDTH - 40)
    rationale_lines = rationale_lines[:7]
    y = 118
    for line in rationale_lines:
        draw.text((18, y), line, font=font(16), fill=(60, 60, 60))
        y += 23
    y += 8
    draw.text((18, y), "完整原始 instruction（黄色仅表示模型建议，不能自动通过）：",
              font=font(20), fill=(20, 20, 20))
    y += 34
    source = event["instruction_text"]
    selected_spans = [
        (segment["char_start"], segment["char_end_exclusive"])
        for segment in event["deterministic_segments"]
        if segment["segment_id"] in selected
    ]
    available = TOP_HEIGHT - y - 62
    selected_font = None
    ranges = None
    for size in range(24, 15, -1):
        candidate_font = font(size)
        candidate_ranges = exact_wrap_ranges(
            draw, source, candidate_font, INSTRUCTION_WIDTH - 40)
        if len(candidate_ranges) * (size + 9) <= available:
            selected_font = candidate_font
            ranges = candidate_ranges
            line_height = size + 9
            break
    if selected_font is None or ranges is None:
        raise RuntimeError("complete instruction did not fit")
    for start, end in ranges:
        x = 18.0
        for index in range(start, end):
            character = source[index]
            char_width = draw.textlength(character, font=selected_font)
            highlighted = any(a <= index < b for a, b in selected_spans)
            if highlighted and not character.isspace():
                draw.rectangle((x - 1, y - 1, x + max(char_width, 4) + 1,
                                y + selected_font.size + 5),
                               fill=(255, 230, 105))
            draw.text((x, y), character, font=selected_font,
                      fill=(20, 20, 20))
            x += char_width
        y += line_height
    warning_y = TOP_HEIGHT - 48
    draw.rectangle((0, warning_y - 5, INSTRUCTION_WIDTH, TOP_HEIGHT),
                   fill=(255, 240, 222))
    draw.text((18, warning_y),
              "必须由人确认/改正子句；MLLM 不是 RxR 官方对齐或训练真值。",
              font=font(19), fill=(135, 45, 25))
    return image, {
        "source_characters": len(source),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "rendered_line_count": len(ranges),
        "font_size_px": selected_font.size,
        "selected_spans": selected_spans,
        "lossless_wrap_verified": True,
    }


def timeline_panel(event, media_by_id, proposal):
    image = Image.new("RGB", (BOARD_WIDTH, TIMELINE_HEIGHT), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.text((14, 7),
              "按时间排列的参考路线抽样帧｜绿色外框=MLLM 引用｜彩色内框=P/D1/D2/D3",
              font=font(20), fill=(255, 255, 255))
    evidence = set(proposal["evidence_frame_ids"])
    roles = {value: key.upper() for key, value in
             event["causal_frame_roles"].items()}
    role_colors = {
        "PRE_REVEAL": (170, 170, 170),
        "D1": (255, 210, 0),
        "D2": (255, 140, 0),
        "D3": (230, 55, 55),
    }
    for index, frame_id in enumerate(event["sequence_frame_ids"]):
        row, col = divmod(index, TILES_PER_ROW)
        x = 9 + col * TILE_SLOT_WIDTH
        y = 48 + row * 181
        record = media_by_id[frame_id]
        frame = Image.open(ROOT / record["path"]).convert("RGB")
        frame = ImageOps.fit(frame, (TILE_IMAGE_SIZE, TILE_IMAGE_SIZE),
                             method=Image.Resampling.LANCZOS)
        image.paste(frame, (x + 8, y + 3))
        if frame_id in evidence:
            draw.rectangle((x + 4, y - 1, x + TILE_IMAGE_SIZE + 12,
                            y + TILE_IMAGE_SIZE + 7),
                           outline=(35, 220, 75), width=4)
        role = roles.get(frame_id)
        if role:
            draw.rectangle((x + 9, y + 4, x + TILE_IMAGE_SIZE + 7,
                            y + TILE_IMAGE_SIZE + 2),
                           outline=role_colors[role], width=4)
        short = frame_id.split("_")[-1]
        label = short + (("  " + role) if role else "")
        draw.text((x + 7, y + TILE_IMAGE_SIZE + 10), label,
                  font=font(14), fill=(245, 245, 245))
    return image


def media_record(path: Path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "pixels": list(Image.open(path).size),
    }


def main() -> int:
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit("input SHA drift: " + str(path))
    acceptance = json.loads(MLLM_ACCEPTANCE.read_text())
    if acceptance.get("status") != "PASS" or \
            acceptance.get("events_passed") != 35:
        raise SystemExit("MLLM acceptance is not 35/35 PASS")
    local = json.loads(LOCAL_PACKET.read_text())
    mllm_input = json.loads(MLLM_INPUT.read_text())
    local_by_event = {row["event_id"]: row for row in local["rows"]}
    media_by_id = {item["frame_id"]: item
                   for item in mllm_input["media_manifest"]}
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    rows, board_manifest = [], []
    for event in mllm_input["events"]:
        event_id = event["event_id"]
        local_row = local_by_event[event_id]
        result_path = MLLM_DIR / "proposals" / f"{event_id}.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "VALID_MLLM_PROPOSAL":
            raise SystemExit(event_id + ": proposal is not valid")
        proposal = result["proposal"]
        old_board = Image.open(ROOT / local_row["private_review_board"]) \
            .convert("RGB")
        if old_board.size != (1920, TOP_HEIGHT):
            raise SystemExit(event_id + ": old board geometry drift")
        evidence = old_board.crop((0, 0, EVIDENCE_WIDTH, TOP_HEIGHT))
        instruction, instruction_render = instruction_panel(event, proposal)
        top = Image.new("RGB", (BOARD_WIDTH, TOP_HEIGHT))
        top.paste(evidence, (0, 0))
        top.paste(instruction, (EVIDENCE_WIDTH, 0))
        timeline = timeline_panel(event, media_by_id, proposal)
        board = Image.new("RGB", (BOARD_WIDTH, BOARD_HEIGHT))
        board.paste(top, (0, 0))
        board.paste(timeline, (0, TOP_HEIGHT))
        board_path = MEDIA_DIR / ("%03d_%s_cr4_review.jpg" %
                                  (event["row_order"], event_id))
        part = board_path.with_suffix(".jpg.part")
        board.save(part, format="JPEG", quality=94, subsampling=0)
        os.replace(part, board_path)
        board_record = media_record(board_path)
        board_manifest.append(board_record)
        segment_text = {item["segment_id"]: item["text"]
                        for item in event["deterministic_segments"]}
        selected_texts = [segment_text[item]
                          for item in proposal["selected_segment_ids"]]
        row = {
            "row_order": event["row_order"],
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "instruction_sha256": event["instruction_sha256"],
            "source_local_review_board": local_row["private_review_board"],
            "source_local_review_board_sha256": sha256_file(
                ROOT / local_row["private_review_board"]),
            "cr4_review_board": board_record["path"],
            "cr4_review_board_sha256": board_record["sha256"],
            "instruction_render": instruction_render,
            "mllm_result_path": str(result_path.relative_to(ROOT)),
            "mllm_result_sha256": sha256_file(result_path),
            "mllm_model": result["provider_response_metadata"]["model"],
            "mllm_status": proposal["status"],
            "mllm_selected_segment_ids": proposal["selected_segment_ids"],
            "mllm_selected_clause_texts": selected_texts,
            "mllm_evidence_frame_ids": proposal["evidence_frame_ids"],
            "mllm_confidence": proposal["confidence"],
            "mllm_proposal_is_ground_truth": False,
            "reviewed": False,
            "reviewer_id": None,
            "review_timestamp": None,
            "clause_alignment_decision": None,
            "human_selected_segment_ids": None,
            "instruction_clause": None,
            "branch_dependent_instruction": None,
            "target_branch_matches_instruction": None,
            "causal_reveal_confirmed": None,
            "semantic_track_confirmed": None,
            "cost_expiry_interpretation_confirmed": None,
            "candidate_valid": None,
            "rejection_reason": None,
            "reviewer_notes": None,
        }
        rows.append(row)
    rows.sort(key=lambda item: item["row_order"])
    if len(rows) != 35 or len({row["event_id"] for row in rows}) != 35:
        raise SystemExit("review row cardinality failure")
    output = {
        "packet": "MF2-CR4 MLLM-assisted human verification",
        "revision": "phase0c-cr4-human-review/1",
        "status": "PASS_PENDING_HUMAN_VERIFICATION",
        "correctness_revision": {
            "path": str(CR4.relative_to(ROOT)),
            "sha256": EXPECTED[CR4],
        },
        "inputs": {str(path.relative_to(ROOT)): expected
                   for path, expected in EXPECTED.items()},
        "row_count": 35,
        "scene_count": len({row["scene_id"] for row in rows}),
        "reviewed_true_count": 0,
        "human_fields": HUMAN_FIELDS,
        "human_fields_prefilled": False,
        "board_contract": {
            "pixels": [BOARD_WIDTH, BOARD_HEIGHT],
            "left": "pinned P,D1,D2,D3 and local-map evidence",
            "right": "complete exact instruction with MLLM proposal highlight",
            "bottom": "chronological route-frame sample; MLLM and causal frames bordered",
            "proposal_is_ground_truth": False,
            "full_instruction_visible": True,
        },
        "rows": rows,
        "board_manifest": board_manifest,
        "board_count": len(board_manifest),
        "board_total_bytes": sum(item["bytes"] for item in board_manifest),
        "network_calls_made_by_builder": 0,
        "private_distribution_authorized": False,
        "verified_language_reveal_events": 0,
        "human_verification_required": True,
        "training_authorized": False,
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    csv_fields = [
        "row_order", "event_id", "episode_id", "scene_id",
        "instruction_sha256", "cr4_review_board",
        "cr4_review_board_sha256", "mllm_model", "mllm_status",
        "mllm_selected_segment_ids", "mllm_selected_clause_texts",
        "mllm_evidence_frame_ids", "mllm_confidence",
    ] + HUMAN_FIELDS
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            for key in ("mllm_selected_segment_ids",
                        "mllm_selected_clause_texts",
                        "mllm_evidence_frame_ids"):
                value[key] = json.dumps(value[key], ensure_ascii=False,
                                        separators=(",", ":"))
            for key in HUMAN_FIELDS:
                if value[key] is None:
                    value[key] = ""
            writer.writerow(value)
    GUIDE.write_text("""# MF2-CR4 人工审核说明（中文）

## 你要判断什么

每张图右侧黄色部分是 `qwen3.8-max` 建议与当前 `P–D3` 局部过程对应的
instruction 子句。它只是帮你从长指令中定位，不是答案，也不能自动通过。

## 每项审核步骤

1. 先读右侧完整 instruction。黄色文字必须是该事件局部过程对应的最小连续片段。
2. 看底部时间序列。绿色框是模型用作理由的帧，彩色内框是 P、D1、D2、D3。
3. 再看左上四张因果画面和左下局部地图。地图是离线审核材料，不代表智能体已看到。
4. 填 `clause_alignment_decision`：
   - `CONFIRM`：黄色片段唯一且正确；
   - `REVISE`：模型定位不对，但你能用 1–3 个相邻 S 编号唯一改正；
   - `AMBIGUOUS`：存在多个合理位置；
   - `REJECT`：没有匹配片段或视觉证据不足。
5. `CONFIRM` 时把模型 S 编号复制到 `human_selected_segment_ids`；`REVISE` 时填你改正的
   1–3 个相邻 S 编号。复制对应原文到 `instruction_clause`，禁止改写。
6. 仅在子句定位唯一后填写冻结的前五个布尔判断。只有定位唯一、前五项全为 true 且你
   确信时，`candidate_valid=true`；否则必须为 false 并写 `rejection_reason`。
7. 填 `reviewed=true`、审核者匿名 ID 和 ISO-8601 时间戳。

## 什么算通过

单项通过要求：子句为 `CONFIRM` 或唯一的 `REVISE`，前五个冻结判断全为 true，且
`candidate_valid=true`。`AMBIGUOUS`、`REJECT`、任何不确定或缺字段都不通过。

本表 35 项必须全部审核。即使人工表完成，也仍需 CR4 规定的两个独立 VLM 审计和最终
汇总关卡；本包本身不授权训练。
""", encoding="utf-8")
    LLM_PROMPT.write_text("""# Fixed MF2-CR4 independent VLM audit prompt

You are an independent audit track, not a human annotator and not the MLLM
clause proposer. Inspect all 35 private CR4 review boards without seeing the
human review or any other audit result. Do not access val_unseen, test, or
test_challenge data.

For each row, independently verify whether the highlighted proposed clause is
the unique smallest one-to-three adjacent exact instruction segments grounding
the P--D3 local route window. The proposal is untrusted guidance: reject or
correct it when the complete instruction and chronological frames disagree.
Then independently judge the six frozen CR3 booleans. Any ambiguity or
insufficient evidence must yield candidate_valid=false. Preserve reviewer_type
=model, exact model/version, prompt SHA, packet SHA, timestamp, raw output, and
all rejection reasons. Never describe this audit as human review or official
RxR alignment, and never authorize training.
""", encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "rows": len(rows),
        "boards": len(board_manifest),
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
        "csv_sha256": sha256_file(CSV),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
