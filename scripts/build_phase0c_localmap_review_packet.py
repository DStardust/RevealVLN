#!/usr/bin/env python3
"""Build a readable local-map review packet without replacing v1 evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
HABSIM = ROOT / "third_party/habitat-sim"
for item in (str(SCRIPTS), str(HABSIM)):
    if item not in sys.path:
        sys.path.insert(0, item)

import build_phase0c_language_review_packet as v1_builder


V1 = ROOT / "artifacts/phase0/phase0c_language_review_35/PHASE0C_LANGUAGE_REVIEW_35.json"
AUTO_RAW = ROOT / "artifacts/runtime/phase0_correctness/AUTOMATIC_SEMANTIC_CANDIDATE_GATE.json"
RXR_TRAIN = Path(v1_builder.RXR_TRAIN)
MP3D = Path(v1_builder.MP3D)
OUT_DIR = ROOT / "artifacts/phase0/phase0c_language_review_35_v2_localmap"
MEDIA_DIR = OUT_DIR / "private_media"
OUT = OUT_DIR / "PHASE0C_LANGUAGE_REVIEW_35_V2_LOCALMAP.json"
EXPECTED = {
    V1: "b97f546d454d09a57c21153adc55bc02c30a4c694b07cd925091fac0b07a6784",
    AUTO_RAW: "13797692e69847392b572f17f0559f36b685ec84b10051fc14c9f26c13ad2f7b",
}
MAP_SIZE = 640
FRAME_SIZE = 280
LEGEND_WIDTH = 480
M_PER_PIXEL = 0.04
MIN_SPAN_M = 6.0
MAP_PADDING_M = 1.25
HFOV_DEG = 63.0
FONT_PATH = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
FONT_SHA256 = "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b"
EVIDENCE_WIDTH = FRAME_SIZE * 4
BOARD_HEIGHT = FRAME_SIZE + MAP_SIZE
INSTRUCTION_WIDTH = 800
BOARD_WIDTH = EVIDENCE_WIDTH + INSTRUCTION_WIDTH
COLORS = {
    "P": (150, 150, 150),
    "D1": (0, 210, 255),
    "D2": (0, 140, 255),
    "D3": (30, 30, 230),
    "candidate": (230, 125, 30),
    "target": (30, 180, 30),
    "route": (180, 120, 40),
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def font(size: int):
    return ImageFont.truetype(str(FONT_PATH), size=size)


def bgr_to_rgb(color):
    return tuple(reversed(tuple(int(value) for value in color)))


def draw_text_bgr(image, text, position, size, fill=(255, 255, 255)):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    draw.text(position, text, font=font(size), fill=fill)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def wrap_by_pixels(draw, text, text_font, max_width):
    """Losslessly wrap text; returned lines reconstruct the exact source."""
    lines, current = [], ""
    for character in text:
        current += character
        width = draw.textbbox((0, 0), current, font=text_font)[2]
        if len(current) <= 1 or width <= max_width:
            continue
        # Prefer a word boundary and retain the whitespace at the end of the
        # previous line so concatenating lines reproduces the source exactly.
        break_at = max(current.rfind(" ", 0, len(current) - 1),
                       current.rfind("\t", 0, len(current) - 1))
        if break_at > 0:
            lines.append(current[:break_at + 1])
            current = current[break_at + 1:]
        else:
            lines.append(current[:-1])
            current = character
    if current or not lines:
        lines.append(current)
    if "".join(lines) != text:
        raise RuntimeError("instruction wrapping changed source text")
    return lines


def build_pathfinder(scene: str):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(MP3D / scene / (scene + ".glb"))
    sim_cfg.gpu_device_id = int(os.environ.get("PHASE0C_MAP_GPU", "1"))
    # Habitat-Sim 0.1.7 can only recompute the accepted 0.88m/0.18m-agent
    # navmesh when scene geometry is renderer-backed. A tiny unused sensor
    # keeps that geometry available; all RGB review frames remain the pinned v1
    # files and are not rerendered here.
    sensor = habitat_sim.SensorSpec()
    sensor.uuid = "unused_rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [1, 1]
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(
        sim_cfg, [agent_cfg]))
    navmesh = MP3D / scene / (scene + ".navmesh")
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError("navmesh load failed: " + scene)
    return sim


def forward_vector(heading: float) -> np.ndarray:
    return np.asarray([-math.sin(heading), -math.cos(heading)],
                      dtype=np.float64)


def draw_dashed_line(image, start, end, color, thickness=2,
                     dash=10, gap=7):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    if distance < 1e-6:
        return
    direction = (end - start) / distance
    cursor = 0.0
    while cursor < distance:
        a = start + direction * cursor
        b = start + direction * min(distance, cursor + dash)
        cv2.line(image, tuple(np.rint(a).astype(int)),
                 tuple(np.rint(b).astype(int)), color, thickness, cv2.LINE_AA)
        cursor += dash + gap


def alpha_polygon(image, points, color, alpha=0.14):
    overlay = image.copy()
    cv2.fillPoly(overlay, [np.asarray(points, dtype=np.int32)], color,
                 cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def put_label(image, text, point, color, scale=0.55):
    x, y = point
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(image, (x - 3, y - height - 5),
                  (x + width + 3, y + baseline + 2), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def draw_compact_marker(image, center, text, color, radius=12):
    """Draw a legible exact-position marker without an overlapping label box."""
    center = tuple(int(value) for value in center)
    cv2.circle(image, center, radius + 2, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    scale = 0.42 if len(text) == 1 else 0.34
    thickness = 2
    (width, height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(image, text,
                (center[0] - width // 2, center[1] + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)


def local_map(pathfinder, trace, prefixes, target, prefix_records):
    labels = ["P", "D1", "D2", "D3"]
    states = [trace[prefixes[key]] for key in
              ("pre_reveal", "d1", "d2", "d3")]
    agent_xyz = np.asarray([state["position"] for state in states],
                           dtype=np.float64)
    candidates = []
    records_by_prefix = {int(record["prefix_index"]): record
                         for record in prefix_records}
    for key in ("d1", "d2", "d3"):
        record = records_by_prefix.get(int(prefixes[key]))
        if record is None or record.get("selected_candidate_endpoint_q") is None:
            candidates.append(None)
        else:
            candidates.append(np.asarray(
                record["selected_candidate_endpoint_q"], dtype=np.float64))
    target_start = np.asarray(target["directed_start_q"], dtype=np.float64)
    target_end = np.asarray(target["directed_end_q"], dtype=np.float64)
    route_start = max(0, int(prefixes["pre_reveal"]) - 3)
    route_end = min(len(trace), int(prefixes["d3"]) + 6)
    route_xyz = np.asarray([trace[index]["position"]
                            for index in range(route_start, route_end)],
                           dtype=np.float64)

    relevant = [agent_xyz[:, [0, 2]], route_xyz[:, [0, 2]],
                target_start[[0, 2]][None], target_end[[0, 2]][None]]
    relevant.extend(candidate[[0, 2]][None] for candidate in candidates
                    if candidate is not None)
    relevant_xz = np.vstack(relevant)
    low, high = relevant_xz.min(0), relevant_xz.max(0)
    center = (low + high) / 2.0
    span = max(MIN_SPAN_M, float(np.max(high - low)) + 2 * MAP_PADDING_M)
    world_low = center - span / 2.0
    event_height = float(np.median(agent_xyz[:, 1]))

    bounds = pathfinder.get_bounds()
    xs = world_low[0] + (np.arange(MAP_SIZE) + 0.5) / MAP_SIZE * span
    zs = world_low[1] + (np.arange(MAP_SIZE) + 0.5) / MAP_SIZE * span
    cols = np.floor((xs - float(bounds[0][0])) / M_PER_PIXEL).astype(int)
    rows = np.floor((zs - float(bounds[0][2])) / M_PER_PIXEL).astype(int)

    def sample_height(height):
        topdown = pathfinder.get_topdown_view(
            meters_per_pixel=M_PER_PIXEL, height=float(height)).astype(np.uint8)
        sampled = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        local_valid_cols = (cols >= 0) & (cols < topdown.shape[1])
        local_valid_rows = (rows >= 0) & (rows < topdown.shape[0])
        if local_valid_cols.any() and local_valid_rows.any():
            sampled[np.ix_(local_valid_rows, local_valid_cols)] = topdown[
                np.ix_(rows[local_valid_rows], cols[local_valid_cols])]
        return sampled

    sampled = sample_height(event_height)
    auxiliary_heights = []
    for height in (float(target_start[1]), float(target_end[1])):
        if (abs(height - event_height) > 0.35 and
                all(abs(height - prior) > 0.2 for prior in auxiliary_heights)):
            auxiliary_heights.append(height)
    auxiliary = np.zeros_like(sampled)
    for height in auxiliary_heights:
        auxiliary = np.maximum(auxiliary, sample_height(height))
    panel = np.empty((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    panel[:] = (48, 48, 48)
    panel[(auxiliary > 0) & (sampled == 0)] = (125, 112, 96)
    panel[sampled > 0] = (224, 224, 224)
    auxiliary_edge = cv2.morphologyEx(
        auxiliary, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    panel[(auxiliary_edge > 0) & (sampled == 0)] = (70, 60, 50)
    edge = cv2.morphologyEx(sampled, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), dtype=np.uint8))
    panel[edge > 0] = (10, 10, 10)

    def xy(point):
        x = int(round((float(point[0]) - world_low[0]) / span *
                      (MAP_SIZE - 1)))
        y = int(round((float(point[2]) - world_low[1]) / span *
                      (MAP_SIZE - 1)))
        return (x, y)

    # A short local route context, not the full episode trajectory.
    route_points = [xy(point) for point in route_xyz]
    for start, end in zip(route_points[:-1], route_points[1:]):
        cv2.line(panel, start, end, COLORS["route"], 2, cv2.LINE_AA)

    # Draw causal 63-degree view sectors and heading arrows.
    view_radius_m = min(1.15, span * 0.16)
    for label, state, color in zip(labels, states,
                                   [COLORS[item] for item in labels]):
        center_px = np.asarray(xy(state["position"]), dtype=np.float64)
        sector = [center_px]
        for offset in np.linspace(-HFOV_DEG / 2.0, HFOV_DEG / 2.0, 15):
            direction = forward_vector(
                float(state["heading"]) + math.radians(float(offset)))
            endpoint = np.asarray(state["position"], dtype=np.float64).copy()
            endpoint[0] += direction[0] * view_radius_m
            endpoint[2] += direction[1] * view_radius_m
            sector.append(np.asarray(xy(endpoint), dtype=np.float64))
        alpha_polygon(panel, sector, color, alpha=0.11)
        heading_endpoint = np.asarray(state["position"], dtype=np.float64).copy()
        direction = forward_vector(float(state["heading"]))
        heading_endpoint[0] += direction[0] * view_radius_m
        heading_endpoint[2] += direction[1] * view_radius_m
        cv2.arrowedLine(panel, tuple(center_px.astype(int)),
                        xy(heading_endpoint), color, 3, cv2.LINE_AA,
                        tipLength=0.22)
    # Exact positions remain visible even when 0.25m steps cluster tightly.
    for marker, state, color in zip(
            ("P", "1", "2", "3"), states,
            [COLORS[item] for item in labels]):
        draw_compact_marker(panel, xy(state["position"]), marker, color, 10)

    # Each causal model proposal is connected to its actual observation pose.
    for index, candidate in enumerate(candidates, start=1):
        if candidate is None:
            continue
        origin = xy(agent_xyz[index])  # D1/D2/D3 correspond to indices 1..3.
        endpoint = xy(candidate)
        draw_dashed_line(panel, origin, endpoint, COLORS["candidate"], 2)
        draw_compact_marker(panel, endpoint, "C%d" % index,
                            COLORS["candidate"], 11)

    # The fixed directed semantic exit is green; its height delta is explicit.
    start_px, end_px = xy(target_start), xy(target_end)
    draw_dashed_line(panel, start_px, end_px, COLORS["target"], 4,
                     dash=14, gap=8)
    cv2.arrowedLine(panel, start_px, end_px, COLORS["target"], 4,
                    cv2.LINE_AA, tipLength=0.16)
    cv2.circle(panel, end_px,
               max(8, int(round(float(target["tube_radius_m"]) /
                                span * MAP_SIZE))),
               COLORS["target"], 2, cv2.LINE_AA)
    # B is the beginning of the fixed post-turn branch segment.  It is not the
    # pre-reveal camera pose P and must not be labelled as an ambiguous "S".
    draw_compact_marker(panel, start_px, "B", COLORS["target"], 12)
    draw_compact_marker(panel, end_px, "T", COLORS["target"], 12)

    # Scale and coordinate legend.
    scale_px = int(round(1.0 / span * MAP_SIZE))
    cv2.line(panel, (24, MAP_SIZE - 28), (24 + scale_px, MAP_SIZE - 28),
             (255, 255, 255), 4, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (MAP_SIZE - 1, 34), (15, 15, 15), -1)
    title = "局部导航网格  高度=%.2f米" % event_height
    if auxiliary_heights:
        title += "  |  蓝灰=分支其他高度"
    panel = draw_text_bgr(panel, title, (10, 3), 20)
    panel = draw_text_bgr(panel, "1米", (24, MAP_SIZE - 59), 18)

    geometry = {
        "event_height_m": round(event_height, 6),
        "target_start_height_m": round(float(target_start[1]), 6),
        "target_end_height_m": round(float(target_end[1]), 6),
        "branch_height_delta_m": round(float(target_end[1] - target_start[1]),
                                         6),
        "pre_reveal_to_branch_start_distance_m": round(float(
            np.linalg.norm(agent_xyz[0] - target_start)), 6),
        "local_square_span_m": round(span, 6),
        "source_topdown_meters_per_pixel": M_PER_PIXEL,
        "navigable_fraction_in_local_crop": round(float(sampled.mean()), 6),
        "auxiliary_branch_heights_m": [round(value, 6)
                                        for value in auxiliary_heights],
        "union_navigable_fraction_in_local_crop": round(float(
            np.maximum(sampled, auxiliary).mean()), 6),
        "candidate_endpoints_drawn": sum(item is not None
                                          for item in candidates),
        "view_hfov_deg": HFOV_DEG,
        "route_context_prefix_range": [route_start, route_end - 1],
    }
    return panel, geometry


def frame_with_title(path: Path, title: str):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot decode frame: " + str(path))
    image = cv2.resize(image, (FRAME_SIZE, FRAME_SIZE),
                       interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(image, (0, 0), (FRAME_SIZE - 1, 34), (0, 0, 0), -1)
    return draw_text_bgr(image, title, (8, 2), 22)


def legend_panel(row, geometry):
    canvas = Image.new("RGB", (LEGEND_WIDTH, MAP_SIZE), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, LEGEND_WIDTH - 1, 42), fill=(18, 18, 18))
    draw.text((12, 6), "局部地图图例", font=font(24), fill=(255, 255, 255))
    entries = [
        ("P", COLORS["P"], "观察前位置"),
        ("D1", COLORS["D1"], "第一次因果观察"),
        ("D2", COLORS["D2"], "第二次因果观察"),
        ("D3", COLORS["D3"], "第三次因果观察"),
        ("C1-C3", COLORS["candidate"], "模型候选落点"),
        ("B→T", COLORS["target"], "固定目标分支（B=分支入口）"),
    ]
    y = 74
    for symbol, color, description in entries:
        draw.ellipse((15, y - 12, 29, y + 2), fill=bgr_to_rgb(color))
        draw.text((40, y - 18), symbol, font=font(18), fill=(20, 20, 20))
        draw.text((145, y - 18), description, font=font(18),
                  fill=(45, 45, 45))
        y += 34
    notes = [
        "浅灰：事件高度的可通行区域",
        "蓝灰：分支其他高度的可通行区域（若有）",
        "彩色扇形：真实 63° 视野与朝向",
        "蓝色虚线：观察位置到候选落点",
        "P：观察前相机位置；不是整条路线起点",
        "绿色箭头：目标分支入口 B → 目标 T",
        "P 到 B 的三维直线距离：%.2f 米" %
            geometry["pre_reveal_to_branch_start_distance_m"],
        "地图仅用于离线审核，绝不是模型输入",
        "分支高度变化：%+0.2f 米" % geometry["branch_height_delta_m"],
        "局部地图跨度：%.2f 米" % geometry["local_square_span_m"],
        "事件：%s" % row["event_id"],
        "场景：%s" % row["scene_id"],
    ]
    y += 10
    for note in notes:
        draw.text((14, y - 15), note, font=font(16), fill=(35, 35, 35))
        y += 27
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def instruction_panel(row):
    width = INSTRUCTION_WIDTH
    canvas = Image.new("RGB", (width, BOARD_HEIGHT),
                       (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, 48), fill=(18, 18, 18))
    draw.text((14, 7), "完整导航指令（私有，以下文本逐字显示）",
              font=font(25), fill=(255, 255, 255))
    source = row["instruction_text_for_private_review"]
    selected_size, lines = None, None
    for size in (29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18):
        selected_font = font(size)
        candidate = wrap_by_pixels(draw, source, selected_font, width - 36)
        line_height = size + 10
        if 62 + len(candidate) * line_height <= BOARD_HEIGHT - 8:
            selected_size, lines = size, candidate
            break
    if selected_size is None:
        raise RuntimeError("full instruction does not fit fixed panel")
    selected_font = font(selected_size)
    line_height = selected_size + 10
    y = 62
    for line in lines:
        draw.text((18, y), line, font=selected_font, fill=(25, 25, 25))
        y += line_height
    metadata = {
        "source_text_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_characters": len(source),
        "rendered_lines": len(lines),
        "font_size_px": selected_size,
        "panel_width_px": INSTRUCTION_WIDTH,
        "panel_height_px": BOARD_HEIGHT,
        "lossless_wrap_verified": "".join(lines) == source,
    }
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR), metadata


def media_record(path: Path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> int:
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit("input SHA drift: " + str(path))
    if sha256_file(FONT_PATH) != FONT_SHA256:
        raise SystemExit("project-local Chinese font SHA drift")
    v1 = json.loads(V1.read_text())
    auto = json.loads(AUTO_RAW.read_text())
    auto_map = {event["provisional_event_id"]: event
                for event in auto["events"]}
    episode_ids = {str(row["episode_id"]) for row in v1["rows"]}
    with gzip.open(RXR_TRAIN, "rt") as handle:
        episodes = {str(item["episode_id"]): item
                    for item in json.load(handle)["episodes"]
                    if str(item["episode_id"]) in episode_ids}
    if len(episodes) != len({row["episode_id"] for row in v1["rows"]}):
        raise SystemExit("episode closure failed")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    output_rows, manifest = [], []
    by_scene = {}
    for row in v1["rows"]:
        by_scene.setdefault(row["scene_id"], []).append(row)
    for scene, scene_rows in sorted(by_scene.items()):
        sim = build_pathfinder(scene)
        try:
            trace_cache = {}
            for row in scene_rows:
                episode_id = str(row["episode_id"])
                if episode_id not in trace_cache:
                    trace_cache[episode_id] = v1_builder.build_lowlevel_trace(
                        sim.pathfinder, episodes[episode_id])
                trace = trace_cache[episode_id]
                event_id = row["event_id"]
                panel, geometry = local_map(
                    sim.pathfinder, trace, row["causal_prefixes"],
                    row["target_exit_region"],
                    auto_map[event_id]["prefix_records"])
                order = int(row["row_order"])
                local_path = MEDIA_DIR / ("%03d_%s_local_map.jpg" %
                                          (order, event_id))
                if not cv2.imwrite(str(local_path), panel,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError("local map write failed")

                titles = ["P：观察前", "D1：第一次观察", "D2：第二次观察",
                          "D3：第三次观察"]
                frames = [frame_with_title(ROOT / path, title)
                          for path, title in zip(row["private_media"], titles)]
                top = np.concatenate(frames, axis=1)
                legend = legend_panel(row, geometry)
                bottom = np.concatenate([panel, legend], axis=1)
                if bottom.shape[1] != top.shape[1]:
                    raise RuntimeError("review board width mismatch")
                full_instruction, instruction_render = instruction_panel(row)
                evidence = np.concatenate([top, bottom], axis=0)
                if (evidence.shape[:2] != (BOARD_HEIGHT, EVIDENCE_WIDTH)
                        or full_instruction.shape[:2] !=
                            (BOARD_HEIGHT, INSTRUCTION_WIDTH)):
                    raise RuntimeError("wide review board geometry mismatch")
                board = np.concatenate([evidence, full_instruction], axis=1)
                board_path = MEDIA_DIR / ("%03d_%s_review_board.jpg" %
                                          (order, event_id))
                if not cv2.imwrite(str(board_path), board,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError("review board write failed")
                manifest.extend(media_record(ROOT / path)
                                for path in row["private_media"])
                manifest.extend([media_record(local_path),
                                 media_record(board_path)])
                updated = dict(row)
                updated["private_contact_sheet_v1_deprecated"] = row[
                    "private_contact_sheet"]
                updated["private_local_map"] = str(local_path.relative_to(ROOT))
                updated["private_review_board"] = str(board_path.relative_to(ROOT))
                updated["private_contact_sheet"] = str(board_path.relative_to(ROOT))
                updated["local_map_geometry"] = geometry
                updated["instruction_render"] = instruction_render
                updated["annotation_status"] = "PENDING_HUMAN_REVIEW_V2_LOCALMAP"
                output_rows.append(updated)
        finally:
            sim.close()

    output_rows.sort(key=lambda row: int(row["row_order"]))
    if len(output_rows) != 35 or len(manifest) != 210:
        raise SystemExit("output cardinality failure")
    if len({item["path"] for item in manifest}) != 210:
        raise SystemExit("duplicate media reference")
    output = {
        "packet": "MF2-CR3 Phase0C language review 35 v2 local map",
        "revision": "phase0c-language-review-localmap/4-wide-instruction",
        "status": "PASS_PENDING_HUMAN_REVIEW",
        "supersedes_for_review": {
            "path": str(V1.relative_to(ROOT)),
            "sha256": sha256_file(V1),
            "reason": "v1 global-route panel lacked navigability, pose "
                      "heading, view cones, candidate links and a legend",
            "v1_evidence_deleted_or_modified": False,
        },
        "inputs": {str(path.relative_to(ROOT)): sha256_file(path)
                   for path in EXPECTED},
        "selection_unchanged": True,
        "row_count": len(output_rows),
        "scene_count": len({row["scene_id"] for row in output_rows}),
        "reviewed_true_count": 0,
        "all_rows_pending": True,
        "human_fields": v1["human_fields"],
        "human_fields_prefilled": False,
        "map_contract": {
            "background": "local navmesh slice at median causal-pose height",
            "route": "short local context only",
            "poses": "P,D1,D2,D3 with physical heading and 63-degree wedge",
            "candidates": "C1-C3 actual automatic selected endpoints joined "
                          "to their causal observation poses",
            "target": "fixed directed post-turn branch entry B to target T; "
                       "B is distinct from pre-reveal camera pose P",
            "offline_only": True,
            "model_input": False,
            "human_visible_guidance_language": "zh-CN",
            "llm_prompt_language": "en",
            "full_instruction_untruncated": True,
            "review_board_pixels": [BOARD_WIDTH, BOARD_HEIGHT],
        },
        "font_provenance": {
            "path": str(FONT_PATH.relative_to(ROOT)),
            "sha256": FONT_SHA256,
            "license": "SIL Open Font License 1.1",
            "provenance": "assets/fonts/PROVENANCE.md",
        },
        "rows": output_rows,
        "media_manifest": manifest,
        "media_reference_count": len(manifest),
        "new_media_file_count": 70,
        "media_total_bytes": sum(item["bytes"] for item in manifest),
        "non_conclusions": {
            "human_validated_events": 0,
            "hybrid_review_completed": False,
            "full_gate6_pass": False,
            "training_authorized": False,
            "distribution_authorized": False,
        },
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    guide = OUT_DIR / "REVIEW_GUIDE_ZH.md"
    guide.write_text("""# MF2-CR3 局部俯视图审核说明

- `P`：观察前位置；`D1/D2/D3`：三次因果观察位置。
- 彩色扇形：每个位置真实可见的 63° 视野；箭头是朝向。
- `C1/C2/C3`：每次观察后自动前端提出的候选落点；蓝色虚线连接
  产生它的观察位置。
- `P`：因果观察窗口开始前的相机位置，不是整条路线的起点。
- 绿色 `B -> T`：离线定义的目标语义出口；`B` 是参考路线
  转向后的目标分支入口，并不要求与 `P` 重合；绿色圆圈
  是 1 米匹配区域。高度变化单独列出，避免把楼梯两层误画成同一层。
- 浅灰区域：事件高度处可通行的 navmesh；深灰不是可通行区域。
- 蓝灰区域：若分支跨楼层，显示目标分支其他高度的可通行区域；它只用于
  说明楼梯/高度连接，不与事件层混为同一平面。
- 棕色线：事件附近的短参考路线，仅提供局部上下文。
- 审核板最右侧为完整导航指令。生成器会逐字符无损换行并校验 SHA-256，
  不允许省略或截断。

判断 `causal_reveal_confirmed` 时只能使用上方 P/D1/D2/D3 真实画面。
局部地图仅帮助判断候选是否对应同一个 TARGET，不是模型输入。

## 六项人工判定

1. `branch_dependent_instruction`：指令是否确实要求走该分支。
2. `target_branch_matches_instruction`：绿色 B→T 是否与指令要求一致。
3. `causal_reveal_confirmed`：P 中不清楚，D1–D3 中才变得可行动。
4. `semantic_track_confirmed`：C1–C3 是否为同一个语义出口。
5. `cost_expiry_interpretation_confirmed`：延迟后返回成本上升/预算不足的
   语义是否合理；精确数值已由机器关卡验证。
6. `candidate_valid`：只有前五项全部为真且你有把握时才为真。

任何不确定项都填 `false` 并填写 `rejection_reason`。字段名保留英文仅为
验证器兼容；人工说明、图例和流程均使用中文。Qwen/Codex 使用另行固定的
英文提示词。
""")
    print(json.dumps({
        "status": output["status"], "rows": len(output_rows),
        "scenes": output["scene_count"],
        "media_references": len(manifest), "new_media_files": 70,
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUT),
        "guide": str(guide.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
