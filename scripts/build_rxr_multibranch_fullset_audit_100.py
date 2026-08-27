#!/usr/bin/env python3
"""Build a portable, deterministic 100-event full-set human audit packet."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/mnt/daiyang/vla").resolve()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_phase0c_cr5_human_review as visual  # noqa: E402
from phase0c_oracle_lowlevel_probe import (  # noqa: E402
    absolute_heading,
    signed_delta,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
CONTROLLER = V2 / "RXR_MULTIBRANCH_CONTROLLER_EXECUTION_V2.json"
ANALYSIS = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
MEDIA = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_MEDIA_MANIFEST_V2.json"
LANGUAGE = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
TX = V2 / "RXR_MULTIBRANCH_TX_V2_GATE.json"
FEATURE = V2 / "RXR_MULTIBRANCH_FEATURE_GATE_V2.json"
FONT = ROOT / "artifacts/fonts/NotoSansSC-wght.ttf"
FONT_LICENSE = ROOT / "artifacts/fonts/NotoSansSC-OFL.txt"

OUT = BASE / "multibranch_fullset_audit_100"
BOARDS = OUT / "boards"
CAUSAL = OUT / "causal"
SELECTION = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_SELECTION.json"
MANIFEST = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_MANIFEST.json"
TEMPLATE = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_TEMPLATE.jsonl"
REVIEWER = OUT / "RXR_MULTIBRANCH_FULLSET_AUDIT_100_REVIEWER.html"
GUIDE = OUT / "审核说明.md"

RANK_SALT = "revealnav-mf2-fullset-human-audit-100-v1"
TWO_BRANCH_QUOTAS = {"train": 43, "development": 10, "gold": 16}
BRANCH_COLORS = [
    (15, 150, 80),
    (168, 65, 190),
    (20, 135, 205),
    (225, 120, 25),
]
CANVAS = (3600, 1900)
RIGHT_X = 2625


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    if (not path.is_file() or path.is_symlink()
            or ROOT not in path.resolve().parents):
        raise RuntimeError("missing or unsafe source: " + str(path))
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               sort_keys=True) + "\n")
    os.replace(part, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(value, encoding="utf-8")
    os.replace(part, path)


def rank(event_id: str, cohort: str) -> str:
    return hashlib.sha256(
        (RANK_SALT + "|" + cohort + "|" + event_id).encode()
    ).hexdigest()


def unique(rows):
    result = {}
    for row in rows:
        event_id = row["event_id"]
        if event_id in result:
            raise RuntimeError("duplicate event: " + event_id)
        result[event_id] = row
    return result


def select(index_rows):
    mandatory = sorted(
        (row for row in index_rows if row["candidate_branch_count"] >= 3),
        key=lambda row: (rank(row["event_id"], "mandatory"),
                         row["event_id"]),
    )
    if len(mandatory) != 31:
        raise RuntimeError("unexpected multi-branch population")
    selected = list(mandatory)
    for split, quota in TWO_BRANCH_QUOTAS.items():
        remaining = [
            row for row in index_rows
            if row["candidate_branch_count"] == 2 and row["split"] == split
        ]
        scene_counts = Counter(
            row["scene_id"] for row in mandatory if row["split"] == split
        )
        chosen = []
        while len(chosen) < quota:
            row = min(
                remaining,
                key=lambda value: (
                    scene_counts[value["scene_id"]],
                    rank(value["event_id"], "two-branch-" + split),
                    value["event_id"],
                ),
            )
            remaining.remove(row)
            chosen.append(row)
            scene_counts[row["scene_id"]] += 1
        selected.extend(chosen)
    if len(selected) != 100 or len({row["event_id"] for row in selected}) != 100:
        raise RuntimeError("audit selection closure failure")
    return selected


def link_media(source: Path, destination: Path, expected_sha: str,
               expected_bytes: int) -> None:
    if (not source.is_file() or source.is_symlink()
            or ROOT not in source.resolve().parents
            or source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_sha):
        raise RuntimeError("causal media drift: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (not destination.is_file() or destination.is_symlink()
                or destination.stat().st_size != expected_bytes
                or sha256_file(destination) != expected_sha):
            raise RuntimeError("existing audit media drift: " + str(destination))
        return
    part = destination.with_name(destination.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError("stale part: " + str(part))
    os.link(source, part)
    os.replace(part, destination)


def branch_rows(geometry):
    return [geometry["target"], *geometry["alternatives"]]


def support_view(event, branch):
    q = np.asarray(branch["Q"], dtype=float)
    endpoint = branch.get("B_star_at_1m") or branch.get("B_i_at_1m")
    heading = absolute_heading(q, np.asarray(endpoint, dtype=float))
    route = float(event["positions"]["Q"]["route_forward_heading_rad"])
    relative = math.degrees(signed_delta(heading, route))
    views = event["positions"]["Q"]["views"]
    return min(views, key=lambda row: abs(
        (float(row["relative_yaw_deg"]) - relative + 180) % 360 - 180
    ))["view_id"]


def local_map(event, geometry, size=(1300, 730)):
    image = Image.new("RGB", size, (248, 249, 251))
    draw = ImageDraw.Draw(image)
    branches = branch_rows(geometry)
    paths = [[np.asarray(row["position_q"], dtype=float)
              for row in branch["path_samples"]] for branch in branches]
    points = [point for path in paths for point in path]
    points.extend(np.asarray(event["positions"][role]["position_q"], dtype=float)
                  for role in ("A", "Q", "D"))
    xs, zs = [p[0] for p in points], [p[2] for p in points]
    margin = 0.8
    xmin, xmax = min(xs) - margin, max(xs) + margin
    zmin, zmax = min(zs) - margin, max(zs) + margin
    width, height = size
    scale = min((width - 100) / max(1e-6, xmax - xmin),
                (height - 135) / max(1e-6, zmax - zmin))
    cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2

    def project(point):
        return (width / 2 + (float(point[0]) - cx) * scale,
                height / 2 - (float(point[2]) - cz) * scale)

    q = np.asarray(geometry["target"]["Q"], dtype=float)
    included = excluded = 0
    for triangle in visual.load_navmesh_triangles(event["scene_id"]):
        centroid = triangle.mean(axis=0)
        if math.hypot(centroid[0] - q[0], centroid[2] - q[2]) > 4.5:
            continue
        if max(abs(float(point[1] - q[1])) for point in triangle) > 0.35:
            excluded += 1
            continue
        draw.polygon([project(point) for point in triangle],
                     fill=(235, 238, 242), outline=(211, 216, 222))
        included += 1
    for index, (path, color) in enumerate(zip(paths, BRANCH_COLORS), 1):
        projected = [project(point) for point in path]
        draw.line(projected, fill=color, width=10, joint="curve")
        for point in projected[1:-1]:
            draw.ellipse((point[0] - 4, point[1] - 4,
                          point[0] + 4, point[1] + 4), fill=color)
        for point, label in ((projected[2], "B%d" % index),
                             (projected[-1], "T%d" % index)):
            draw.rectangle((point[0] - 7, point[1] - 7,
                            point[0] + 7, point[1] + 7), fill=color)
            draw.text((point[0] + 9, point[1] - 10), label,
                      font=visual.font(20), fill=color,
                      stroke_width=2, stroke_fill=(255, 255, 255))
    for role, color in (("A", (36, 110, 190)), ("Q", (20, 20, 20)),
                        ("D", (230, 125, 30))):
        point = project(event["positions"][role]["position_q"])
        draw.ellipse((point[0] - 10, point[1] - 10,
                      point[0] + 10, point[1] + 10), fill=color,
                     outline=(255, 255, 255), width=3)
        draw.text((point[0] + 12, point[1] - 16), role,
                  font=visual.font(22), fill=color,
                  stroke_width=2, stroke_fill=(255, 255, 255))
    draw.text((24, 15), "局部俯视图：全部候选分支（Q同层）",
              font=visual.font(30), fill=(25, 30, 36))
    legend = "绿色=目标；紫/蓝/橙=其他候选；B=约1m；T=约1.75m"
    draw.text((24, height - 38), legend, font=visual.font(21),
              fill=(55, 60, 68))
    draw.text((width - 365, 18), "同层三角%d；隐藏异层%d" %
              (included, excluded), font=visual.font(19), fill=(90, 95, 103))
    return image


def elevation_plot(geometry, size=(1245, 325)):
    image = Image.new("RGB", size, (248, 249, 251))
    draw = ImageDraw.Draw(image)
    branches = branch_rows(geometry)
    qy = float(geometry["target"]["Q"][1])
    values = [float(row["position_q"][1]) - qy
              for branch in branches for row in branch["path_samples"]]
    ymin, ymax = min(-0.25, min(values) - 0.1), max(0.25, max(values) + 0.1)
    left, right, top, bottom = 60, 25, 55, 45

    def project(row):
        x = left + float(row["offset_m"]) / 1.75 * (size[0] - left - right)
        y_value = float(row["position_q"][1]) - qy
        y = top + (ymax - y_value) / (ymax - ymin) * (
            size[1] - top - bottom)
        return x, y

    zero = top + ymax / (ymax - ymin) * (size[1] - top - bottom)
    draw.line((left, zero, size[0] - right, zero), fill=(170, 175, 183), width=2)
    for branch, color in zip(branches, BRANCH_COLORS):
        draw.line([project(row) for row in branch["path_samples"]],
                  fill=color, width=7, joint="curve")
    draw.text((18, 10), "全部分支高度剖面（相对Q）", font=visual.font(28),
              fill=(25, 30, 36))
    return image


def build_board(event, geometry, language_row, decisive_clause_ids, split):
    canvas = Image.new("RGB", CANVAS, (18, 21, 25))
    draw = ImageDraw.Draw(canvas)
    title = "%s | 全分支独立审计 | %d个候选" % (
        event["event_id"], len(branch_rows(geometry)))
    draw.text((28, 18), title, font=visual.font(42), fill=(245, 247, 250),
              stroke_width=1, stroke_fill=(0, 0, 0))

    panorama = visual.safe_image(event["positions"]["Q"]["contact_sheet"])
    canvas.paste(visual.fit(panorama, 1245, 820), (25, 90))
    draw.text((38, 102), "Q决策中心360°（仅供离线检查候选完整性）",
              font=visual.font(25), fill=(255, 255, 255),
              stroke_width=3, stroke_fill=(0, 0, 0))

    branches = branch_rows(geometry)
    card_height = 255 if len(branches) >= 3 else 385
    for index, (branch, color) in enumerate(zip(branches, BRANCH_COLORS)):
        y = 90 + index * (card_height + 16)
        view_id = support_view(event, branch)
        view = visual.safe_view_image(event, view_id)
        canvas.paste(visual.fit(view, 505, card_height), (1295, y))
        draw.rectangle((1295, y, 1800, y + card_height), outline=color, width=7)
        role = "目标" if index == 0 else "候选%d" % (index + 1)
        draw.text((1310, y + 10), "%s %s %s" %
                  (role, branch["branch_id"], view_id),
                  font=visual.font(26), fill=color,
                  stroke_width=3, stroke_fill=(255, 255, 255))
        visual.draw_box(draw, (1818, y, 2570, y + card_height),
                        (245, 247, 250), outline=color, width=4)
        text = "%s / %s\n%s" % (
            branch["horizontal_direction"], branch["vertical_motion"],
            branch["visual_descriptor"])
        visual.text_block(draw, (1840, y + 20), text, visual.font(22),
                          (35, 42, 50), 700, line_spacing=6,
                          max_lines=7 if card_height >= 300 else 5)

    canvas.paste(local_map(event, geometry), (25, 955))
    draw.rectangle((25, 955, 1325, 1685), outline=(80, 85, 92), width=2)
    draw.text((1360, 958), "时间顺序：A → Q → D（路线前向63°）",
              font=visual.font(28), fill=(235, 238, 242))
    for index, role in enumerate(("A", "Q", "D")):
        image = visual.safe_view_image(event, role + "_V00")
        x = 1360 + index * 410
        canvas.paste(visual.fit(image, 385, 330), (x, 1010))
        draw.rectangle((x, 1010, x + 385, 1340),
                       outline=(130, 135, 143), width=2)
        draw.text((x + 10, 1295), {"A": "A 接近", "Q": "Q 决策中心",
                                  "D": "D 离开"}[role],
                  font=visual.font(23), fill=(255, 230, 80),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    canvas.paste(elevation_plot(geometry), (1360, 1370))

    panel = (RIGHT_X, 90, CANVAS[0] - 22, CANVAS[1] - 24)
    visual.draw_box(draw, panel, (245, 247, 250),
                    outline=(105, 112, 122), width=3)
    x, y, width = RIGHT_X + 25, 110, CANVAS[0] - RIGHT_X - 75
    draw.text((x, y), "人工审核（不是训练标签）", font=visual.font(33),
              fill=(25, 30, 36))
    y += 52
    visual.text_block(draw, (x, y),
                      "离线全景用于核对是否漏分支；网页下方严格时间帧用于核对在线因果性。",
                      visual.font(21), (70, 75, 83), width, max_lines=3)
    y += 78
    segments = {row["segment_id"]: row["text"]
                for row in event["deterministic_segments"]}
    relevant = ["%s: %s" % (key, segments[key]) for key in decisive_clause_ids
                if key in segments]
    draw.text((x, y), "判定性原文子句", font=visual.font(28),
              fill=(155, 80, 10))
    y += 42
    y = visual.text_block(draw, (x, y), "\n".join(relevant) or "（无）",
                          visual.font(21), (90, 55, 18), width,
                          line_spacing=5, max_lines=8)
    y += 10
    draw.text((x, y), "完整 instruction", font=visual.font(28),
              fill=(30, 75, 135))
    y += 42
    instruction_font = visual.font(20)
    for size in range(20, 14, -1):
        candidate = visual.font(size)
        instruction_font = candidate
        if len(visual.wrap_lines(draw, event["instruction_text"],
                                 candidate, width)) <= 24:
            break
    y = visual.text_block(draw, (x, y), event["instruction_text"],
                          instruction_font, (30, 36, 43), width,
                          line_spacing=5, max_lines=24)
    y += 12
    draw.line((x, y, x + width, y), fill=(190, 195, 202), width=2)
    y += 12
    ids = [branch["branch_id"] for branch in branches]
    summary = (
        "候选全集：%s；绿色%s为自动目标。\n"
        "Reveal区间：P%04d–P%04d；网页黄框为确认帧。" %
        (", ".join(ids), geometry["target"]["branch_id"],
         language_row["reveal_interval"][0],
         language_row["reveal_interval"][1])
    )
    y = visual.text_block(draw, (x, y), summary, visual.font(21),
                          (40, 80, 58), width, max_lines=4)
    y += 12
    draw.text((x, y), "五项通过标准", font=visual.font(28),
              fill=(145, 35, 35))
    y += 42
    checklist = (
        "① 360°范围没有遗漏明显的可执行出口；\n"
        "② 所列分支均可走且不是来路/关闭门/重复/短假分叉；\n"
        "③ 指令能在全部候选中唯一选择绿色目标；\n"
        "④ Q确需决策且A→Q→D时序合理；\n"
        "⑤ 黄色确认帧前的63°历史已足够，不依赖未来。\n"
        "全是=ACCEPT；任一明确否=REJECT；看不清=AMBIGUOUS。"
    )
    visual.text_block(draw, (x, y), checklist, visual.font(21),
                      (70, 30, 30), width, max_lines=11)
    return canvas


def template_row(event_id: str):
    return {
        "reviewer_id": None,
        "reviewer_type": "HUMAN",
        "event_id": event_id,
        "candidate_set_complete": None,
        "all_candidates_distinct_and_executable": None,
        "instruction_uniquely_selects_target_among_all": None,
        "decision_center_and_temporal_order_reasonable": None,
        "causal_prefix_supports_reveal_without_future_frames": None,
        "final_label": None,
        "reason_codes": [],
        "comment_zh": "",
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>RevealNav 全分支100条审计</title><style>
body{margin:0;background:#11151a;color:#edf1f5;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:#1b222a;padding:9px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,input{font-size:15px;padding:7px 10px}.ok{background:#16854c;color:white}.bad{background:#a53535;color:white}.amb{background:#9a6a10;color:white}
#board{display:block;width:98vw;max-width:3600px;height:auto;margin:8px auto;border:1px solid #4b5561}
#causal{display:flex;gap:7px;overflow-x:auto;padding:8px 14px;background:#20262d}.frame{flex:0 0 auto;text-align:center;color:#cbd3dc}.frame img{height:230px;border:3px solid #56606b}.reveal img{border-color:#29b765}.confirm img{border-color:#ffd34d;border-width:5px}
#controls{padding:9px 15px 24px;display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:8px}#comment{grid-column:1/-1;width:calc(100% - 26px)}.wide{grid-column:1/-1}.muted{color:#aeb8c2}
</style></head><body><header><b id="progress"></b><span id="event"></span>
<label>审核者 <input id="reviewer" value="daiyang" size="12"></label>
<button onclick="move(-1)">← 上一条</button><button onclick="move(1)">下一条 →</button>
<button onclick="exportJsonl()">导出JSONL</button><label><button onclick="document.getElementById('importer').click()">导入进度</button><input id="importer" type="file" hidden></label></header>
<img id="board" alt="full-set review board"><div id="causal"></div><div id="controls">
<button class="ok wide" onclick="accept()">A：五项都通过 → ACCEPT</button>
<button class="bad" onclick="reject(0)">1：候选集合漏掉明显出口</button>
<button class="bad" onclick="reject(1)">2：存在来路/关闭/重复/假分叉</button>
<button class="bad" onclick="reject(2)">3：指令不能唯一选择绿色目标</button>
<button class="bad" onclick="reject(3)">4：Q位置或A→Q→D时序不合理</button>
<button class="bad" onclick="reject(4)">5：确认帧不足或依赖未来</button>
<button class="amb" onclick="ambiguous()">U：证据不足 → AMBIGUOUS</button><button onclick="clearCurrent()">清除本条</button>
<input id="comment" placeholder="可选中文备注"><div class="wide muted">上方360°只做离线候选完整性审核；下方横向63°历史才是在线证据。绿框=Reveal区间，黄框=确认帧。快捷键：A、1–5、U、←、→。</div></div>
<script>const items=__ITEMS__;const keys=['candidate_set_complete','all_candidates_distinct_and_executable','instruction_uniquely_selects_target_among_all','decision_center_and_temporal_order_reasonable','causal_prefix_supports_reveal_without_future_frames'];
const reasons=['CANDIDATE_SET_INCOMPLETE','CANDIDATE_INCOMING_CLOSED_DUPLICATE_OR_SHORT','INSTRUCTION_TARGET_NOT_UNIQUE_AMONG_ALL','DECISION_CENTER_OR_TEMPORAL_ORDER_INVALID','CAUSAL_REVEAL_NEEDS_FUTURE_OR_IS_NOT_SUPPORTED'];let index=0;let labels=JSON.parse(localStorage.getItem('revealnav_fullset100_labels_v1')||'{}');
function base(){return {reviewer_id:document.getElementById('reviewer').value||null,reviewer_type:'HUMAN',event_id:items[index].event_id,candidate_set_complete:null,all_candidates_distinct_and_executable:null,instruction_uniquely_selects_target_among_all:null,decision_center_and_temporal_order_reasonable:null,causal_prefix_supports_reveal_without_future_frames:null,final_label:null,reason_codes:[],comment_zh:document.getElementById('comment').value||''}}
function save(r){labels[r.event_id]=r;localStorage.setItem('revealnav_fullset100_labels_v1',JSON.stringify(labels));render();setTimeout(()=>move(1),120)}
function accept(){let r=base();keys.forEach(k=>r[k]=true);r.final_label='ACCEPT';save(r)}function reject(i){let r=base();r[keys[i]]=false;r.final_label='REJECT';r.reason_codes=[reasons[i]];save(r)}function ambiguous(){let r=base();r.final_label='AMBIGUOUS';r.reason_codes=['INSUFFICIENT_VISUAL_EVIDENCE'];save(r)}
function clearCurrent(){delete labels[items[index].event_id];localStorage.setItem('revealnav_fullset100_labels_v1',JSON.stringify(labels));render()}function move(d){index=Math.max(0,Math.min(items.length-1,index+d));render()}
function render(){let it=items[index],r=labels[it.event_id];document.getElementById('board').src=it.board;document.getElementById('event').textContent=it.event_id+' | '+it.branch_count+'分支 | '+it.scene_id+(r?' | 已标 '+r.final_label:' | 未标');document.getElementById('progress').textContent=(index+1)+' / '+items.length+'（已完成 '+Object.keys(labels).length+'）';document.getElementById('comment').value=r?.comment_zh||'';let box=document.getElementById('causal');box.innerHTML='';for(let f of it.frames){let d=document.createElement('div');d.className='frame'+(f.reveal?' reveal':'')+(f.confirm?' confirm':'');let im=document.createElement('img');im.src=f.src;let cap=document.createElement('div');cap.textContent=f.frame_id+(f.confirm?' 确认':'');d.append(im,cap);box.append(d)}}
function exportJsonl(){let rows=items.filter(x=>labels[x.event_id]).map(x=>JSON.stringify(labels[x.event_id]));let blob=new Blob([rows.join('\n')+(rows.length?'\n':'')],{type:'application/jsonl'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='daiyang_fullset100.jsonl';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importer').addEventListener('change',async e=>{let text=await e.target.files[0].text();for(let line of text.split(/\r?\n/)){if(!line.trim())continue;let r=JSON.parse(line);if(items.some(x=>x.event_id===r.event_id))labels[r.event_id]=r}localStorage.setItem('revealnav_fullset100_labels_v1',JSON.stringify(labels));render()});document.addEventListener('keydown',e=>{if(['comment','reviewer'].includes(document.activeElement.id))return;let k=e.key.toLowerCase();if(k==='a')accept();else if(k==='u')ambiguous();else if(['1','2','3','4','5'].includes(k))reject(Number(k)-1);else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});render();</script></body></html>'''


GUIDE_TEXT = """# RevealNav 全分支100条独立审计说明

解压完整审计包后，直接打开 `RXR_MULTIBRANCH_FULLSET_AUDIT_100_REVIEWER.html`。不要只复制 HTML；`boards/` 和 `causal/` 必须与它放在同一目录。

每条依次检查：

1. 360° 离线全景中，候选集合是否遗漏明显、可执行且与任务相关的出口。
2. 所列全部分支是否真实可走、彼此不同，并且不是来路、关闭门、重复视角或很快汇合的假分叉。
3. 完整指令是否能在全部候选中唯一选择绿色目标分支。
4. Q 是否确实需要做选择，A→Q→D 的时间顺序是否合理。
5. 网页下方严格按时间排列的 63° 前向历史，是否在黄色确认帧已经足够支持 Reveal，不依赖未来画面。

五项都明确为“是”才按 A 接受；任一明确为“否”按 1–5 拒绝；证据不足按 U。完成后点击“导出JSONL”，把 `daiyang_fullset100.jsonl` 放回本目录。该审计不自动授权训练。
"""


def main() -> int:
    sources = (INPUTS, INDEX, GEOMETRY, CONTROLLER, ANALYSIS, MEDIA,
               LANGUAGE, TX, FEATURE, FONT, FONT_LICENSE)
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path)
                     for path in sources}
    index_doc = load(INDEX)
    tx_doc = load(TX)
    feature_doc = load(FEATURE)
    if (index_doc.get("status") != "FEATURE_AND_TX_GENERATION_REQUIRED"
            or tx_doc.get("status") != "MULTIBRANCH_TX_PASS"
            or feature_doc.get("status") != "FEATURE_GATE_PASS_AUDIT_REQUIRED"):
        raise SystemExit("full-set audit prerequisites failed")
    inputs = unique(load(INPUTS)["events"])
    geometry = unique(load(GEOMETRY)["events"])
    controller = unique(load(CONTROLLER)["events"])
    analysis = unique(load(ANALYSIS)["events"])
    language = unique(load(LANGUAGE)["events"])
    media_doc = load(MEDIA)
    media = {}
    for row in media_doc["media_manifest"]:
        media[(row["episode_id"], row["frame_id"])] = row

    selected = select(index_doc["records"])
    rows = []
    for review_index, index_row in enumerate(selected, 1):
        event_id = index_row["event_id"]
        event = inputs[event_id]
        geo = geometry[event_id]
        control = controller[event_id]
        causal = analysis[event_id]
        lang = language[event_id]
        branch_ids = index_row["candidate_branch_ids"]
        if not (
            geo["status"] == "GEOMETRY_PASS_CONTROLLER_REQUIRED"
            and control["status"] == "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
            and causal["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
            and lang["status"] == "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"
            and geo["candidate_branch_ids"] == branch_ids
            and control["candidate_branch_ids"] == branch_ids
            and causal["candidate_branch_ids"] == branch_ids
            and len(branch_rows(geo)) == len(branch_ids)
        ):
            raise RuntimeError("selected event closure drift: " + event_id)
        confirmation = lang["confirmation_prefix"]
        tested = next(row for row in lang["tested_prefixes"]
                      if row["prefix_index"] == confirmation)
        evidence_path = ROOT / tested["path"]
        if sha256_file(evidence_path) != tested["sha256"]:
            raise RuntimeError("language evidence drift: " + event_id)
        evidence = load(evidence_path)
        request = evidence["request_evidence"]
        response = evidence.get("parsed_response") or {}
        if (request["maximum_media_prefix"] != confirmation
                or request["future_frames_in_request"] != 0
                or request["panoramas_in_request"] != 0
                or response.get("selected_branch_id") !=
                index_row["target_branch_id"]):
            raise RuntimeError("causal evidence boundary drift: " + event_id)
        linked = []
        for frame_id, expected_sha in zip(
                request["frame_ids"], request["media_sha256"]):
            record = media[(index_row["episode_id"], frame_id)]
            if record["sha256"] != expected_sha or record["hfov_deg"] != 63.0:
                raise RuntimeError("causal frame drift: " + event_id)
            destination = CAUSAL / event_id / (frame_id + ".jpg")
            link_media(ROOT / record["path"], destination,
                       record["sha256"], record["bytes"])
            prefix = record["prefix_index"]
            linked.append({
                "frame_id": frame_id,
                "prefix_index": prefix,
                "path": str(destination.relative_to(ROOT)),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "hfov_deg": record["hfov_deg"],
                "reveal_interval_member": (
                    lang["reveal_interval"][0] <= prefix
                    <= lang["reveal_interval"][1]
                ),
                "confirmation_frame": prefix == confirmation,
            })
        image = build_board(event, geo, lang,
                            response.get("decisive_clause_ids", []),
                            index_row["split"])
        board_path = BOARDS / (event_id + "_review.jpg")
        board_path.parent.mkdir(parents=True, exist_ok=True)
        part = board_path.with_name(board_path.name + ".part")
        image.save(part, format="JPEG", quality=90, subsampling=0,
                   optimize=True)
        os.replace(part, board_path)
        rows.append({
            "review_index": review_index,
            "cohort": ("MANDATORY_ALL_THREE_OR_FOUR_BRANCH"
                       if index_row["candidate_branch_count"] >= 3 else
                       "SCENE_BALANCED_TWO_BRANCH_" + index_row["split"].upper()),
            "event_id": event_id,
            "episode_id": index_row["episode_id"],
            "scene_id": index_row["scene_id"],
            "split": index_row["split"],
            "candidate_branch_ids": branch_ids,
            "candidate_branch_count": len(branch_ids),
            "target_branch_id": index_row["target_branch_id"],
            "reveal_interval": lang["reveal_interval"],
            "confirmation_prefix": confirmation,
            "rank_sha256": rank(event_id, "mandatory" if len(branch_ids) >= 3
                                  else "two-branch-" + index_row["split"]),
            "instruction_sha256": event["instruction_sha256"],
            "language_evidence_path": str(evidence_path.relative_to(ROOT)),
            "language_evidence_sha256": tested["sha256"],
            "board_path": str(board_path.relative_to(ROOT)),
            "board_bytes": board_path.stat().st_size,
            "board_sha256": sha256_file(board_path),
            "board_pixels": list(CANVAS),
            "causal_media": linked,
            "human_status": "PENDING",
            "training_label": False,
        })
        print("%03d/100 %s" % (review_index, event_id), flush=True)

    selection = {
        "schema_version": "revealnav-mf2-fullset-audit-selection/1",
        "status": "SELECTION_FROZEN_BEFORE_HUMAN_LABELING",
        "rank_salt": RANK_SALT,
        "population_count": len(index_doc["records"]),
        "selection_protocol": {
            "mandatory": "all 31 events with three or four branches",
            "two_branch_quotas": TWO_BRANCH_QUOTAS,
            "within_split": (
                "greedy minimum selected scene count, then salted SHA-256 rank"
            ),
        },
        "selected_count": len(rows),
        "counts_by_cohort": dict(Counter(row["cohort"] for row in rows)),
        "counts_by_split": dict(Counter(row["split"] for row in rows)),
        "unique_scenes": len({row["scene_id"] for row in rows}),
        "items": [{key: row[key] for key in (
            "review_index", "cohort", "event_id", "episode_id", "scene_id",
            "split", "candidate_branch_count", "rank_sha256"
        )} for row in rows],
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(SELECTION, selection)
    manifest = {
        "schema_version": "revealnav-mf2-fullset-human-audit-package/1",
        "status": "READY_FOR_FRESH_FULLSET_HUMAN_AUDIT",
        "sources": source_hashes,
        "selection": {"path": str(SELECTION.relative_to(ROOT)),
                      "sha256": sha256_file(SELECTION)},
        "items": rows,
        "counts": {
            "items": len(rows),
            "three_or_four_branch": sum(
                row["candidate_branch_count"] >= 3 for row in rows),
            "two_branch": sum(row["candidate_branch_count"] == 2
                              for row in rows),
            "causal_frames": sum(len(row["causal_media"]) for row in rows),
            "unique_scenes": len({row["scene_id"] for row in rows}),
        },
        "future_frames_in_online_evidence": 0,
        "panoramas_in_online_evidence": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    atomic_text(TEMPLATE, "".join(
        json.dumps(template_row(row["event_id"]), ensure_ascii=False,
                   sort_keys=True) + "\n" for row in rows
    ))
    ui = [{
        "event_id": row["event_id"],
        "scene_id": row["scene_id"],
        "branch_count": row["candidate_branch_count"],
        "board": "boards/" + Path(row["board_path"]).name,
        "frames": [{
            "frame_id": frame["frame_id"],
            "src": "causal/%s/%s.jpg" % (row["event_id"], frame["frame_id"]),
            "reveal": frame["reveal_interval_member"],
            "confirm": frame["confirmation_frame"],
        } for frame in row["causal_media"]],
    } for row in rows]
    atomic_text(REVIEWER, HTML.replace(
        "__ITEMS__", json.dumps(ui, ensure_ascii=False,
                                separators=(",", ":"))))
    atomic_text(GUIDE, GUIDE_TEXT)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "output": str(OUT.relative_to(ROOT)),
        "training_authorized": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
