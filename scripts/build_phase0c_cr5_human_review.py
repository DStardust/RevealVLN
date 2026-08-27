#!/usr/bin/env python3
"""Build the first human-review packet for controller-verified CR5 events."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/daiyang/vla")
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", str(ROOT / "third_party/habitat-sim")))
if str(HABSIM) not in sys.path:
    sys.path.insert(0, str(HABSIM))
import habitat_sim  # noqa: E402


BASE = ROOT / "artifacts/phase0/phase0c_cr5_preflight/multiview_branch"
INPUT = BASE / "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
PRESCREEN = BASE / "CR5_MULTIVIEW_MAIN_AGENT_PRESCREEN_V2.json"
GEOMETRY = BASE / "CR5_DIRECTED_GEOMETRY_PREFLIGHT.json"
CONTROLLER = BASE / "CR5_CONTROLLER_EXECUTION_PREFLIGHT.json"
PROPOSALS = BASE / "proposals_v2"
FONT = ROOT / "artifacts/fonts/NotoSansSC-wght.ttf"
FONT_LICENSE = ROOT / "artifacts/fonts/NotoSansSC-OFL.txt"
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_human_review_v1"
BOARD_DIR = OUT_DIR / "boards"
MANIFEST = OUT_DIR / "CR5_HUMAN_REVIEW_MANIFEST.json"
TEMPLATE = OUT_DIR / "CR5_HUMAN_REVIEW_TEMPLATE.jsonl"

EXPECTED = {
    INPUT: "3d3a1d4ce468c8a54a5a61b96f340a415bad8357442ae242b0cf6b595a12f7fe",
    PRESCREEN: "e14f1c5e61e0f725ae94fd9599455a0e32f30626400964aceb9395b3ccaad5d3",
    GEOMETRY: "92a461a5cebfe84c53bce211bd3c78bec59f8aaa0d2be73b46e814b5bcb374f0",
    CONTROLLER: "3dd638f90b9199b643c0129110ecebb7355effc5c83a467dcb3b1ed691ffe311",
    FONT: "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da",
    FONT_LICENSE: "1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9",
}

CANVAS = (3000, 1800)
RIGHT_X = 2050
TARGET_COLOR = (15, 150, 80)
ALT_COLOR = (170, 55, 190)
APPROACH_COLOR = (36, 110, 190)
DEPART_COLOR = (230, 125, 30)

PRESCREEN_TIER = {
    "ep34121_hv01": ("STRONG", "卧室白门与玻璃出口清楚分离，指令唯一选择白门。"),
    "ep34121_hv03": ("STRONG", "平台存在两个开门与下行楼梯，目标房间有白沙发/玻璃出口线索。"),
    "ep34121_hv05": ("BORDERLINE", "阳台、卫生间与来路可区分，但需人工确认卫生间是否构成任务相关备选。"),
    "ep41233_hv02": ("STRONG", "楼梯向下、卧室门和沿栏杆走廊三路清楚，修复了旧 B/T 反向问题。"),
    "ep43805_hv01": ("STRONG", "起居室通向门厅与厨房的两个开口清楚。"),
    "ep43805_hv03": ("BORDERLINE", "门厅有小走廊、拱门和楼梯；需确认目标小走廊不是短死端。"),
    "ep43805_hv05": ("BORDERLINE", "健身房门与厨房通道可见，但局部空间较窄，需确认不是同一通道视角。"),
    "ep46758_hv02": ("STRONG", "楼梯、开放起居区与红墙侧廊构成明显多路选择。"),
    "ep56443_hv05": ("STRONG", "拱形走廊与厨房方向清楚；已合并相邻重复候选。"),
    "ep7619_hv05": ("STRONG", "门厅楼梯与两条水平开口清楚；已合并三个时间重复候选。"),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def font(size: int, bold: bool = False):
    # The variable font honors layout consistently; stroke is used for the few
    # headings that need extra emphasis instead of acquiring another file.
    return ImageFont.truetype(str(FONT), size=size)


def safe_image(record):
    path = ROOT / record["path"]
    if (not path.is_file() or path.is_symlink()
            or ROOT.resolve() not in path.resolve().parents
            or sha256_file(path) != record["sha256"]):
        raise RuntimeError("unsafe or drifted review image: " + record["path"])
    return Image.open(path).convert("RGB")


def fit(image: Image.Image, width: int, height: int,
        background=(24, 27, 31)):
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)),
            max(1, int(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(resized, ((width - size[0]) // 2,
                           (height - size[1]) // 2))
    return canvas


def draw_box(draw, box, fill, outline=(65, 70, 78), width=2):
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline,
                           width=width)


def wrap_lines(draw, text: str, used_font, max_width: int):
    result = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            result.append("")
            continue
        words = paragraph.split(" ")
        tokens = words if len(words) > 1 else list(paragraph)
        separator = " " if len(words) > 1 else ""
        current = ""
        for token in tokens:
            candidate = token if not current else current + separator + token
            if draw.textlength(candidate, font=used_font) <= max_width:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = token
        if current:
            result.append(current)
    return result


def text_block(draw, xy, text, used_font, fill, max_width,
               line_spacing=8, max_lines=None):
    x, y = xy
    lines = wrap_lines(draw, text, used_font, max_width)
    if max_lines is not None:
        lines = lines[:max_lines]
    bbox = used_font.getbbox("国Ag")
    line_height = bbox[3] - bbox[1] + line_spacing
    for line in lines:
        draw.text((x, y), line, font=used_font, fill=fill)
        y += line_height
    return y


def view_record(event, view_id: str):
    role = view_id.split("_", 1)[0]
    for record in event["positions"][role]["views"]:
        if record["view_id"] == view_id:
            return record
    raise RuntimeError("view ID absent: " + view_id)


def safe_view_image(event, view_id: str):
    """Load a saved view or crop it from the verified 4x3 contact sheet."""
    record = view_record(event, view_id)
    if "path" in record:
        return safe_image(record)
    role = view_id.split("_", 1)[0]
    contact = event["positions"][role]["contact_sheet"]
    view_ids = contact.get("view_ids")
    if (not isinstance(view_ids, list) or len(view_ids) != 12
            or view_id not in view_ids):
        raise RuntimeError("contact-sheet view closure failure: " + view_id)
    sheet = safe_image(contact)
    if sheet.width % 4 or sheet.height % 3:
        raise RuntimeError("unexpected contact-sheet geometry: " + view_id)
    index = view_ids.index(view_id)
    width, height = sheet.width // 4, sheet.height // 3
    left, top = (index % 4) * width, (index // 4) * height
    return sheet.crop((left, top, left + width, top + height))


def supporting_view(branch):
    values = branch["supporting_view_ids"]
    for prefix in ("Q_", "A_", "D_"):
        for value in values:
            if value.startswith(prefix):
                return value
    raise RuntimeError("branch has no supporting view")


@lru_cache(maxsize=None)
def load_navmesh_triangles(scene: str):
    path = ROOT / (
        "third_party/ETP-R1/data/scene_datasets/mp3d/" + scene + "/"
        + scene + ".navmesh")
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(path)):
        raise RuntimeError("review navmesh load failed")
    raw = np.asarray(pathfinder.build_navmesh_vertices(), dtype=float)
    return raw.reshape(-1, 3, 3)


def local_map(scene: str, event, geometry, size=(1180, 690)):
    image = Image.new("RGB", size, (248, 249, 251))
    draw = ImageDraw.Draw(image)
    q = np.asarray(geometry["target"]["Q"], dtype=float)
    target = [np.asarray(row["position_q"], dtype=float)
              for row in geometry["target"]["path_samples"]]
    alternative = [np.asarray(row["position_q"], dtype=float)
                   for row in geometry["alternative"]["path_samples"]]
    points = target + alternative + [
        np.asarray(event["positions"][role]["position_q"], dtype=float)
        for role in ("A", "Q", "D")]
    xs = [point[0] for point in points]
    zs = [point[2] for point in points]
    margin = 0.75
    xmin, xmax = min(xs) - margin, max(xs) + margin
    zmin, zmax = min(zs) - margin, max(zs) + margin
    width, height = size
    top = 74
    bottom = 54
    left = 55
    right = 35
    scale = min((width - left - right) / max(1e-6, xmax - xmin),
                (height - top - bottom) / max(1e-6, zmax - zmin))
    cx = (xmin + xmax) / 2
    cz = (zmin + zmax) / 2

    def project(point):
        return (
            width / 2 + (float(point[0]) - cx) * scale,
            (top + height - bottom) / 2 - (float(point[2]) - cz) * scale,
        )

    included = excluded = 0
    for triangle in load_navmesh_triangles(scene):
        centroid = triangle.mean(axis=0)
        if math.hypot(centroid[0] - q[0], centroid[2] - q[2]) > 4.5:
            continue
        if max(abs(float(point[1] - q[1])) for point in triangle) > 0.35:
            excluded += 1
            continue
        polygon = [project(point) for point in triangle]
        draw.polygon(polygon, fill=(235, 238, 242),
                     outline=(211, 216, 222))
        included += 1

    def route(values, color, width_px=9):
        projected = [project(point) for point in values]
        draw.line(projected, fill=color, width=width_px, joint="curve")
        for point in projected[1:-1]:
            draw.ellipse((point[0] - 4, point[1] - 4,
                          point[0] + 4, point[1] + 4), fill=color)

    route(target, TARGET_COLOR)
    route(alternative, ALT_COLOR)
    roles = {
        "A": (APPROACH_COLOR, "A 前约1m"),
        "Q": ((25, 25, 25), "Q 决策中心"),
        "D": (DEPART_COLOR, "D 后约1m"),
    }
    for role, (color, label) in roles.items():
        point = project(event["positions"][role]["position_q"])
        draw.ellipse((point[0] - 10, point[1] - 10,
                      point[0] + 10, point[1] + 10), fill=color,
                     outline=(255, 255, 255), width=3)
        draw.text((point[0] + 12, point[1] - 18), label,
                  font=font(23), fill=color,
                  stroke_width=2, stroke_fill=(255, 255, 255))

    labels = [
        (target[2], "B*", TARGET_COLOR),
        (target[-1], "T*", TARGET_COLOR),
        (alternative[2], "Bi", ALT_COLOR),
        (alternative[-1], "Ti", ALT_COLOR),
    ]
    for value, label, color in labels:
        point = project(value)
        draw.rectangle((point[0] - 7, point[1] - 7,
                        point[0] + 7, point[1] + 7), fill=color)
        draw.text((point[0] + 9, point[1] + 2), label, font=font(21),
                  fill=color, stroke_width=2, stroke_fill=(255, 255, 255))

    draw.text((24, 16), "局部俯视图（只画 Q 同层 |Δy|≤0.35m）",
              font=font(31), fill=(25, 30, 36))
    legend = ("绿色=目标分支  紫色=可执行备选  "
              "B=离开决策区约1m  T=下游1.75m")
    draw.text((24, height - 38), legend, font=font(22),
              fill=(55, 60, 68))
    draw.text((width - 360, 19),
              "同层三角 %d；隐藏异层三角 %d" % (included, excluded),
              font=font(20), fill=(95, 100, 108))
    return image


def elevation_plot(geometry, size=(750, 320)):
    image = Image.new("RGB", size, (248, 249, 251))
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 62, 25, 56, 48
    target = geometry["target"]["path_samples"]
    alternative = geometry["alternative"]["path_samples"]
    qy = float(geometry["target"]["Q"][1])
    all_y = [float(row["position_q"][1]) - qy
             for row in target + alternative]
    ymin = min(-0.25, min(all_y) - 0.1)
    ymax = max(0.25, max(all_y) + 0.1)

    def project(row):
        x = left + float(row["offset_m"]) / 1.75 * (
            size[0] - left - right)
        yval = float(row["position_q"][1]) - qy
        y = top + (ymax - yval) / (ymax - ymin) * (
            size[1] - top - bottom)
        return x, y

    zero_y = top + ymax / (ymax - ymin) * (size[1] - top - bottom)
    draw.line((left, zero_y, size[0] - right, zero_y),
              fill=(170, 175, 183), width=2)
    for values, color in ((target, TARGET_COLOR),
                          (alternative, ALT_COLOR)):
        draw.line([project(row) for row in values], fill=color,
                  width=7, joint="curve")
    draw.text((18, 12), "高度剖面（相对 Q）", font=font(29),
              fill=(25, 30, 36))
    draw.text((left, size[1] - 38), "Q 0m", font=font(20),
              fill=(75, 80, 88))
    draw.text((size[0] - 125, size[1] - 38), "T 1.75m",
              font=font(20), fill=(75, 80, 88))
    draw.text((size[0] - 270, 18), "绿=目标  紫=备选",
              font=font(20), fill=(75, 80, 88))
    return image


def build_board(event, proposal, geometry, controller, tier, note,
                causal_review: bool = False):
    canvas = Image.new("RGB", CANVAS, (18, 21, 25))
    draw = ImageDraw.Draw(canvas)
    title = "%s  |  CR5 分支事件审核  |  自动优先级：%s" % (
        event["event_id"], tier)
    draw.text((28, 20), title, font=font(43), fill=(245, 247, 250),
              stroke_width=1, stroke_fill=(0, 0, 0))

    q_panorama = safe_image(event["positions"]["Q"]["contact_sheet"])
    canvas.paste(fit(q_panorama, 1140, 850), (25, 95))
    draw.text((38, 105), "Q：决策中心 360°（12×30°）",
              font=font(27), fill=(255, 255, 255), stroke_width=3,
              stroke_fill=(0, 0, 0))

    branches = {row["branch_id"]: row for row in proposal["branches"]}
    target_branch = branches[geometry["target"]["branch_id"]]
    alt_branch = branches[geometry["alternative"]["branch_id"]]
    selected = [
        ("目标分支", target_branch, TARGET_COLOR, 100),
        ("备选分支", alt_branch, ALT_COLOR, 525),
    ]
    for label, branch, color, y in selected:
        view_id = supporting_view(branch)
        view = safe_view_image(event, view_id)
        canvas.paste(fit(view, 825, 390), (1190, y))
        draw.rectangle((1190, y, 2015, y + 390), outline=color, width=7)
        draw.text((1208, y + 12), "%s  %s  %s" % (
            label, branch["branch_id"], view_id), font=font(30), fill=color,
            stroke_width=3, stroke_fill=(255, 255, 255))
        text_block(draw, (1210, y + 335), branch["visual_descriptor"],
                   font(20), (245, 245, 245), 780, max_lines=2)

    map_image = local_map(event["scene_id"], event, geometry)
    canvas.paste(map_image, (25, 980))
    draw.rectangle((25, 980, 1205, 1670), outline=(80, 85, 92), width=2)

    # A/Q/D are route-relative views.  There is intentionally no S marker:
    # A is an approach sample, not the episode start.
    draw.text((1240, 980), "时间顺序：A → Q → D（均为 V00 路线前向）",
              font=font(29), fill=(235, 238, 242))
    for index, role in enumerate(("A", "Q", "D")):
        image = safe_view_image(event, role + "_V00")
        x = 1240 + index * 255
        canvas.paste(fit(image, 235, 285), (x, 1030))
        draw.rectangle((x, 1030, x + 235, 1315),
                       outline=(130, 135, 143), width=2)
        label = {"A": "A 前约1m", "Q": "Q 决策中心",
                 "D": "D 后约1m"}[role]
        draw.text((x + 8, 1272), label, font=font(23),
                  fill=(255, 230, 80), stroke_width=2,
                  stroke_fill=(0, 0, 0))
    canvas.paste(elevation_plot(geometry), (1240, 1350))

    # Right instruction and decision panel.
    panel = (RIGHT_X, 92, CANVAS[0] - 22, CANVAS[1] - 24)
    draw_box(draw, panel, (245, 247, 250), outline=(105, 112, 122), width=3)
    x = RIGHT_X + 28
    y = 112
    width = CANVAS[0] - x - 50
    draw.text((x, y), "人工审核（不是训练标签）", font=font(36),
              fill=(25, 30, 36))
    y += 60
    y = text_block(draw, (x, y), "自动预筛说明：" + note,
                   font(24), (65, 70, 78), width, max_lines=4)
    y += 12
    draw.line((x, y, x + width, y), fill=(190, 195, 202), width=2)
    y += 16

    segments = {row["segment_id"]: row["text"]
                for row in event["deterministic_segments"]}
    relevant = []
    for segment_id in proposal["action_clause_ids"]:
        relevant.append("%s（动作）: %s" %
                        (segment_id, segments[segment_id]))
    for segment_id in proposal["reveal_clause_ids"]:
        value = "%s（识别）: %s" % (segment_id, segments[segment_id])
        if value not in relevant:
            relevant.append(value)
    draw.text((x, y), "相关原文子句", font=font(30), fill=(160, 85, 10))
    y += 48
    y = text_block(draw, (x, y), "\n".join(relevant), font(23),
                   (95, 55, 15), width, max_lines=8)
    y += 12
    draw.text((x, y), "完整 instruction", font=font(30),
              fill=(30, 75, 135))
    y += 46
    instruction_font = font(22)
    for size in range(22, 15, -1):
        candidate = font(size)
        instruction_font = candidate
        if len(wrap_lines(draw, event["instruction_text"], candidate,
                          width)) <= 22:
            break
    y = text_block(draw, (x, y), event["instruction_text"],
                   instruction_font, (30, 36, 43), width,
                   line_spacing=6)
    y += 14
    draw.line((x, y, x + width, y), fill=(190, 195, 202), width=2)
    y += 15

    evidence = [
        "自动硬门：3D 几何 PASS；目标/备选各 2 次控制器回放 PASS。",
        "目标 %s；备选 %s；夹角 %.1f°；1.75m 分离 %.2fm。" % (
            geometry["target"]["branch_id"],
            geometry["alternative"]["branch_id"],
            geometry["alternative"]["distinctness"][
                "three_dimensional_angle_at_1m_deg"],
            geometry["alternative"]["distinctness"][
                "separation_at_1_75m_m"]),
        "注意：全景只用于离线建标；在线 REE 仍只能看当时及过去的 63° ego-FOV。",
    ]
    y = text_block(draw, (x, y), "\n".join(evidence), font(22),
                   (45, 75, 60), width, max_lines=6)
    y += 12
    draw.text((x, y), "怎样判定", font=font(30), fill=(145, 35, 35))
    y += 46
    checklist = (
        "① 至少两条真正可走且语义不同的出口？\n"
        "② 紫色备选不是来路、关闭的门、同一出口或短暂分叉？\n"
        "③ 完整指令能唯一选择绿色目标？\n"
        "④ Q 确实位于需要选择的共享区域，A→Q→D 顺序合理？\n"
    )
    if causal_review:
        checklist += (
            "⑤ 网页下方63°历史在黄色确认帧已足够，不依赖未来画面？\n"
            "五项都“是”=ACCEPT；任一明确“否”=REJECT；"
            "证据看不清=AMBIGUOUS。")
    else:
        checklist += (
            "四项都“是”=ACCEPT；任一明确“否”=REJECT；"
            "证据看不清=AMBIGUOUS。")
    text_block(draw, (x, y), checklist, font(23), (70, 30, 30), width,
               max_lines=10)
    return canvas


def main() -> int:
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit("input SHA drift: " + str(path))
    inputs = json.loads(INPUT.read_text())
    prescreen = json.loads(PRESCREEN.read_text())
    geometry = json.loads(GEOMETRY.read_text())
    controller = json.loads(CONTROLLER.read_text())
    event_by_id = {row["event_id"]: row for row in inputs["events"]}
    prescreen_by_id = {row["event_id"]: row
                       for row in prescreen["events"]}
    geometry_by_id = {row["event_id"]: row
                      for row in geometry["events"]}
    accepted = [row for row in controller["events"]
                if row["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"]
    if len(accepted) != 10 or set(PRESCREEN_TIER) != {
            row["event_id"] for row in accepted}:
        raise SystemExit("review candidate closure failure")

    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    template_rows = []
    for result in sorted(accepted, key=lambda value: value["event_id"]):
        event_id = result["event_id"]
        event = event_by_id[event_id]
        geometry_row = geometry_by_id[event_id]
        proposal_path = PROPOSALS / (event_id + ".json")
        proposal = json.loads(proposal_path.read_text())[
            "normalized_proposal"]
        tier, note = PRESCREEN_TIER[event_id]
        board = build_board(event, proposal, geometry_row, result,
                            tier, note)
        board_path = BOARD_DIR / (event_id + "_review.jpg")
        board.save(board_path, format="JPEG", quality=93,
                   subsampling=0, optimize=True)
        rows.append({
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "instruction_sha256": event["instruction_sha256"],
            "proposal_sha256": sha256_file(proposal_path),
            "board_path": str(board_path.relative_to(ROOT)),
            "board_bytes": board_path.stat().st_size,
            "board_sha256": sha256_file(board_path),
            "board_pixels": list(CANVAS),
            "main_agent_prescreen_tier": tier,
            "main_agent_prescreen_note_zh": note,
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

    manifest = {
        "manifest": "MF2-CR5 controller-verified human branch review v1",
        "revision": "cr5-human-review/1",
        "status": "READY_FOR_HUMAN_BRANCH_REVIEW",
        "review_scope": (
            "branch validity and instruction target only; causal Reveal timing "
            "remains a separate pending gate"),
        "sources": {
            "multiview_input_sha256": EXPECTED[INPUT],
            "prescreen_sha256": EXPECTED[PRESCREEN],
            "geometry_sha256": EXPECTED[GEOMETRY],
            "controller_sha256": EXPECTED[CONTROLLER],
            "font_sha256": EXPECTED[FONT],
            "font_license_sha256": EXPECTED[FONT_LICENSE],
        },
        "board_count": len(rows),
        "strong_count": sum(row["main_agent_prescreen_tier"] == "STRONG"
                            for row in rows),
        "borderline_count": sum(
            row["main_agent_prescreen_tier"] == "BORDERLINE" for row in rows),
        "items": rows,
        "labels_created": 0,
        "human_reviews_completed": 0,
        "training_authorized": False,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2,
                                   ensure_ascii=False) + "\n")
    TEMPLATE.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in template_rows))
    print(json.dumps({
        "status": manifest["status"],
        "boards": manifest["board_count"],
        "strong": manifest["strong_count"],
        "borderline": manifest["borderline_count"],
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST),
        "template_sha256": sha256_file(TEMPLATE),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
