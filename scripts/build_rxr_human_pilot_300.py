#!/usr/bin/env python3
"""Build the self-contained 300-event RxR human audit packet."""

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


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
ACCEPTANCE = BASE / "RXR_EXPANSION_AUTOMATIC_FILTER_ACCEPTANCE.json"
INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
PRESCREEN = BASE / "branch_factory/RXR_MULTIVIEW_MACHINE_PRESCREEN.json"
GEOMETRY = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
CONTROLLER = BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json"
ANALYSIS = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
MEDIA = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
LANGUAGE = BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json"
AUTHORIZATION = ROOT / (
    "artifacts/upstream/matterport3d/"
    "MP3D_ACCESS_AUTHORIZATION_ATTESTATION.json")
FONT = ROOT / "artifacts/fonts/NotoSansSC-wght.ttf"
FONT_LICENSE = ROOT / "artifacts/fonts/NotoSansSC-OFL.txt"

OUT_DIR = BASE / "human_pilot_300"
BOARD_DIR = OUT_DIR / "boards"
CAUSAL_DIR = OUT_DIR / "causal"
SELECTION = OUT_DIR / "RXR_HUMAN_PILOT_300_SELECTION.json"
MANIFEST = OUT_DIR / "RXR_HUMAN_PILOT_300_MANIFEST.json"
TEMPLATE = OUT_DIR / "RXR_HUMAN_PILOT_300_TEMPLATE.jsonl"
GUIDE = OUT_DIR / "审核说明.md"
REVIEWER = OUT_DIR / "RXR_HUMAN_PILOT_300_REVIEWER.html"

CORE_COUNT = 250
SUPPLEMENT_COUNT = 50
TOTAL_COUNT = CORE_COUNT + SUPPLEMENT_COUNT
RANK_SALT = "revealnav-human-pilot-300-v1"

board.CANVAS = (3000, 1800)
board.RIGHT_X = 2050


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(event_id: str, cohort: str) -> str:
    return hashlib.sha256(
        (RANK_SALT + "|" + cohort + "|" + event_id).encode()).hexdigest()


def load(path: Path):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("missing or unsafe source: " + str(path))
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(value, encoding="utf-8")
    os.replace(part, path)


def select_events(eligible, analysis_by_id):
    ordered = sorted(eligible, key=lambda value: (rank(value, "core"), value))
    core = ordered[:CORE_COUNT]
    remaining = set(ordered[CORE_COUNT:])
    all_scenes = {analysis_by_id[value]["scene_id"] for value in eligible}
    selected_scene_counts = Counter(
        analysis_by_id[value]["scene_id"] for value in core)
    supplement = []

    missing_scenes = sorted(all_scenes - set(selected_scene_counts))
    for scene_id in missing_scenes:
        candidates = [value for value in remaining
                      if analysis_by_id[value]["scene_id"] == scene_id]
        chosen = min(candidates,
                     key=lambda value: (rank(value, "supplement"), value))
        supplement.append(chosen)
        remaining.remove(chosen)
        selected_scene_counts[scene_id] += 1

    while len(supplement) < SUPPLEMENT_COUNT:
        chosen = min(
            remaining,
            key=lambda value: (
                selected_scene_counts[analysis_by_id[value]["scene_id"]],
                rank(value, "supplement"), value))
        supplement.append(chosen)
        remaining.remove(chosen)
        selected_scene_counts[analysis_by_id[chosen]["scene_id"]] += 1

    if (len(core) != CORE_COUNT or len(supplement) != SUPPLEMENT_COUNT
            or set(core) & set(supplement)
            or len(set(core + supplement)) != TOTAL_COUNT
            or set(selected_scene_counts) != all_scenes):
        raise RuntimeError("deterministic pilot selection closure failure")
    return core, supplement


def link_media(source: Path, destination: Path, expected_sha: str,
               expected_bytes: int) -> None:
    if (not source.is_file() or source.is_symlink()
            or ROOT.resolve() not in source.resolve().parents
            or source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_sha):
        raise RuntimeError("causal source drift: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (not destination.is_file() or destination.is_symlink()
                or destination.stat().st_size != expected_bytes
                or sha256_file(destination) != expected_sha):
            raise RuntimeError("existing packet media drift: "
                               + str(destination))
        return
    part = destination.with_name(destination.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError("stale packet part: " + str(part))
    os.link(source, part)
    os.replace(part, destination)


def template_row(event_id: str):
    return {
        "reviewer_id": None,
        "reviewer_type": "HUMAN",
        "event_id": event_id,
        "two_distinct_executable_exits": None,
        "alternative_is_not_incoming_closed_or_duplicate": None,
        "instruction_uniquely_selects_target": None,
        "decision_center_and_temporal_order_are_reasonable": None,
        "causal_prefix_supports_reveal_without_future_frames": None,
        "final_label": None,
        "reason_codes": [],
        "comment_zh": "",
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>RevealNav RxR 300条人工金标审核</title><style>
body{margin:0;background:#11151a;color:#edf1f5;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:#1b222a;padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,input{font-size:15px;padding:7px 10px}.ok{background:#16854c;color:white}.bad{background:#a53535;color:white}.amb{background:#9a6a10;color:white}
#board{display:block;width:98vw;max-width:3000px;height:auto;margin:10px auto;border:1px solid #4b5561}
#causal{display:flex;gap:8px;overflow-x:auto;padding:8px 14px;background:#20262d}.frame{flex:0 0 auto;text-align:center;color:#cbd3dc}.frame img{height:220px;border:3px solid #56606b}.reveal img{border-color:#29b765}.confirm img{border-color:#ffd34d;border-width:5px}
#controls{padding:10px 16px 24px;display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:8px}#comment{grid-column:1/-1;width:calc(100% - 26px)}.wide{grid-column:1/-1}.muted{color:#aeb8c2}
</style></head><body><header><b id="progress"></b><span id="event"></span>
<label>审核者 <input id="reviewer" value="daiyang" size="12"></label>
<button onclick="move(-1)">← 上一条</button><button onclick="move(1)">下一条 →</button>
<button onclick="exportJsonl()">导出 JSONL</button><label><button onclick="document.getElementById('importer').click()">导入进度</button><input id="importer" type="file" hidden></label></header>
<img id="board" alt="branch review board"><div id="causal"></div><div id="controls">
<button class="ok wide" onclick="accept()">A：五项都通过 → ACCEPT</button>
<button class="bad" onclick="reject(0)">1：没有两条不同可走出口</button>
<button class="bad" onclick="reject(1)">2：备选是来路/关闭/重复/假分叉</button>
<button class="bad" onclick="reject(2)">3：指令不能唯一选中绿色目标</button>
<button class="bad" onclick="reject(3)">4：Q位置或A→Q→D时序不合理</button>
<button class="bad" onclick="reject(4)">5：黄色确认帧仍不足或依赖未来画面</button>
<button class="amb" onclick="ambiguous()">U：证据不足 → AMBIGUOUS</button><button onclick="clearCurrent()">清除本条</button>
<input id="comment" placeholder="可选中文备注；REJECT时可补充原因"><div class="wide muted">主图全景仅用于离线分支审计；下方严格时间序列才是在线63°前向证据。绿框=Reveal区间，黄框=K=3确认帧。快捷键：A、1–5、U、←、→。</div></div>
<script>const items=__ITEMS__;const keys=['two_distinct_executable_exits','alternative_is_not_incoming_closed_or_duplicate','instruction_uniquely_selects_target','decision_center_and_temporal_order_are_reasonable','causal_prefix_supports_reveal_without_future_frames'];
const reasons=['NO_TWO_DISTINCT_EXECUTABLE_EXITS','ALTERNATIVE_INCOMING_CLOSED_DUPLICATE_OR_SHORT','INSTRUCTION_TARGET_NOT_UNIQUE','DECISION_CENTER_OR_TEMPORAL_ORDER_INVALID','CAUSAL_REVEAL_NEEDS_FUTURE_OR_IS_NOT_SUPPORTED'];let index=0;let labels=JSON.parse(localStorage.getItem('revealnav_rxr300_labels_v1')||'{}');
function base(){return {reviewer_id:document.getElementById('reviewer').value||null,reviewer_type:'HUMAN',event_id:items[index].event_id,two_distinct_executable_exits:null,alternative_is_not_incoming_closed_or_duplicate:null,instruction_uniquely_selects_target:null,decision_center_and_temporal_order_are_reasonable:null,causal_prefix_supports_reveal_without_future_frames:null,final_label:null,reason_codes:[],comment_zh:document.getElementById('comment').value||''}}
function save(r){labels[r.event_id]=r;localStorage.setItem('revealnav_rxr300_labels_v1',JSON.stringify(labels));render();setTimeout(()=>move(1),150)}
function accept(){let r=base();keys.forEach(k=>r[k]=true);r.final_label='ACCEPT';save(r)}function reject(i){let r=base();r[keys[i]]=false;r.final_label='REJECT';r.reason_codes=[reasons[i]];save(r)}function ambiguous(){let r=base();r.final_label='AMBIGUOUS';r.reason_codes=['INSUFFICIENT_VISUAL_EVIDENCE'];save(r)}
function clearCurrent(){delete labels[items[index].event_id];localStorage.setItem('revealnav_rxr300_labels_v1',JSON.stringify(labels));render()}function move(d){index=Math.max(0,Math.min(items.length-1,index+d));render()}
function render(){let it=items[index],r=labels[it.event_id];document.getElementById('board').src=it.board;document.getElementById('event').textContent=it.event_id+' | '+it.cohort+' | '+it.scene_id+(r?' | 已标 '+r.final_label:' | 未标');document.getElementById('progress').textContent=(index+1)+' / '+items.length+'（已完成 '+Object.keys(labels).length+'）';document.getElementById('comment').value=r?.comment_zh||'';let box=document.getElementById('causal');box.innerHTML='';for(let f of it.frames){let d=document.createElement('div');d.className='frame'+(f.reveal?' reveal':'')+(f.confirm?' confirm':'');let im=document.createElement('img');im.src=f.src;let cap=document.createElement('div');cap.textContent=f.frame_id+(f.confirm?' K=3确认':'');d.append(im,cap);box.append(d)}}
function exportJsonl(){let rows=items.filter(x=>labels[x.event_id]).map(x=>{let r=labels[x.event_id];r.reviewer_id=document.getElementById('reviewer').value||r.reviewer_id;return JSON.stringify(r)});let blob=new Blob([rows.join('\n')+(rows.length?'\n':'')],{type:'application/jsonl'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='daiyang_rxr300.jsonl';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importer').addEventListener('change',async e=>{let text=await e.target.files[0].text();for(let line of text.split(/\r?\n/)){if(!line.trim())continue;let r=JSON.parse(line);if(items.some(x=>x.event_id===r.event_id))labels[r.event_id]=r}localStorage.setItem('revealnav_rxr300_labels_v1',JSON.stringify(labels));render()});document.addEventListener('keydown',e=>{if(['comment','reviewer'].includes(document.activeElement.id))return;let k=e.key.toLowerCase();if(k==='a')accept();else if(k==='u')ambiguous();else if(['1','2','3','4','5'].includes(k))reject(Number(k)-1);else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});render();</script></body></html>'''


def main() -> int:
    sources = [ACCEPTANCE, INPUTS, PRESCREEN, GEOMETRY, CONTROLLER,
               ANALYSIS, MEDIA, LANGUAGE, AUTHORIZATION, FONT, FONT_LICENSE]
    source_sha = {str(path.relative_to(ROOT)): sha256_file(path)
                  for path in sources}
    acceptance = load(ACCEPTANCE)
    inputs = load(INPUTS)
    prescreen = load(PRESCREEN)
    geometry = load(GEOMETRY)
    controller = load(CONTROLLER)
    analysis = load(ANALYSIS)
    media = load(MEDIA)
    language = load(LANGUAGE)
    authorization = load(AUTHORIZATION)
    if (acceptance["status"] != "PASS_READY_FOR_300_HUMAN_PILOT"
            or acceptance["event_floor_pass"] is not True
            or authorization["status"] != "USER_CONFIRMED_AUTHORIZED"):
        raise SystemExit("automatic or authorization prerequisite failed")

    event_by_id = {row["event_id"]: row for row in inputs["events"]}
    prescreen_by_id = {row["event_id"]: row for row in prescreen["events"]}
    geometry_by_id = {row["event_id"]: row for row in geometry["events"]}
    controller_by_id = {row["event_id"]: row for row in controller["events"]}
    analysis_by_id = {row["event_id"]: row for row in analysis["events"]}
    language_by_id = {row["event_id"]: row for row in language["events"]}
    media_by_episode = {}
    for record in media["media_manifest"]:
        media_by_episode.setdefault(record["episode_id"], {})[
            record["frame_id"]] = record

    eligible = list(acceptance["eligible_event_ids"])
    if len(eligible) != 525 or len(set(eligible)) != 525:
        raise SystemExit("unexpected eligible event closure")
    core, supplement = select_events(eligible, analysis_by_id)
    selection_rows = []
    for cohort, values in (("AUDIT_CORE_UNIFORM_250", core),
                           ("SCENE_COVERAGE_SUPPLEMENT_50", supplement)):
        for cohort_index, event_id in enumerate(values, 1):
            selection_rows.append({
                "review_index": len(selection_rows) + 1,
                "cohort": cohort,
                "cohort_index": cohort_index,
                "event_id": event_id,
                "episode_id": analysis_by_id[event_id]["episode_id"],
                "scene_id": analysis_by_id[event_id]["scene_id"],
                "rank_sha256": rank(
                    event_id, "core" if cohort.startswith("AUDIT")
                    else "supplement"),
            })
    selection_output = {
        "manifest": "RevealNav RxR human pilot 300 deterministic selection",
        "revision": "rxr-human-pilot-300-selection/1",
        "status": "SELECTION_FROZEN_BEFORE_HUMAN_LABELING",
        "rank_salt": RANK_SALT,
        "selection_protocol": {
            "audit_core": (
                "first 250 eligible event IDs by SHA-256 rank; this cohort "
                "is the primary automatic-label quality audit"),
            "scene_coverage_supplement": (
                "50 disjoint events selected after the core, first covering "
                "missing scenes and then minimizing current selected scene "
                "count; excluded from unweighted core precision estimates"),
        },
        "eligible_population_count": len(eligible),
        "selected_count": len(selection_rows),
        "selected_scene_count": len({row["scene_id"]
                                     for row in selection_rows}),
        "cohort_counts": dict(Counter(row["cohort"]
                                      for row in selection_rows)),
        "items": selection_rows,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_json(SELECTION, selection_output)

    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    ui_rows = []
    templates = []
    for selected in selection_rows:
        event_id = selected["event_id"]
        event = event_by_id[event_id]
        prescreen_row = prescreen_by_id[event_id]
        geometry_row = geometry_by_id[event_id]
        controller_row = controller_by_id[event_id]
        analysis_row = analysis_by_id[event_id]
        language_row = language_by_id[event_id]
        if (prescreen_row["prescreen_disposition"] not in
                {"TO_DIRECTED_GEOMETRY", "RELOCATE_EARLIER_THEN_3D"}
                or geometry_row["status"] !=
                "GEOMETRY_PASS_CONTROLLER_REQUIRED"
                or controller_row["status"] !=
                "CONTROLLER_PASS_CAUSAL_GATE_REQUIRED"
                or analysis_row["status"] !=
                "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
                or language_row["status"] !=
                "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"):
            raise RuntimeError("selected event gate drift: " + event_id)

        proposal_path = ROOT / prescreen_row["proposal_path"]
        if sha256_file(proposal_path) != prescreen_row["proposal_sha256"]:
            raise RuntimeError("proposal drift: " + event_id)
        proposal = load(proposal_path)["normalized_proposal"]
        confirmation = language_row["confirmation_prefix"]
        tested = next(row for row in language_row["tested_prefixes"]
                      if row["prefix_index"] == confirmation)
        response_path = ROOT / tested["path"]
        if sha256_file(response_path) != tested["sha256"]:
            raise RuntimeError("confirmation evidence drift: " + event_id)
        response = load(response_path)
        request_evidence = response["request_evidence"]
        frame_ids = request_evidence["frame_ids"]
        media_hashes = request_evidence["media_sha256"]
        if (request_evidence["maximum_media_prefix"] != confirmation
                or request_evidence["future_frames_in_request"] != 0
                or request_evidence["panoramas_in_request"] != 0
                or len(frame_ids) != len(media_hashes)):
            raise RuntimeError("causal request boundary drift: " + event_id)

        linked = []
        for frame_id, expected_sha in zip(frame_ids, media_hashes):
            record = media_by_episode[analysis_row["episode_id"]][frame_id]
            if record["sha256"] != expected_sha or record["hfov_deg"] != 63.0:
                raise RuntimeError("causal frame closure drift: " + event_id)
            source = ROOT / record["path"]
            destination = CAUSAL_DIR / event_id / (frame_id + ".jpg")
            link_media(source, destination, record["sha256"], record["bytes"])
            linked.append({
                "frame_id": frame_id,
                "prefix_index": record["prefix_index"],
                "path": str(destination.relative_to(ROOT)),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "hfov_deg": record["hfov_deg"],
                "reveal_interval_member": (
                    language_row["reveal_interval"][0]
                    <= record["prefix_index"]
                    <= language_row["reveal_interval"][1]),
                "confirmation_frame": record["prefix_index"] == confirmation,
            })

        angle = geometry_row["alternative"]["distinctness"][
            "three_dimensional_angle_at_1m_deg"]
        separation = geometry_row["alternative"]["distinctness"][
            "separation_at_1_75m_m"]
        note = (
            "严格在线因果门 K=3 PASS；确认前缀 P%04d。"
            "本图只供离线核验分支，底部时间帧用于核验真实在线可见性。"
            % confirmation)
        image = board.build_board(
            event, proposal, geometry_row, controller_row,
            selected["cohort"], note, causal_review=True)
        board_path = BOARD_DIR / (event_id + "_review.jpg")
        part = board_path.with_name(board_path.name + ".part")
        image.save(part, format="JPEG", quality=90, subsampling=0,
                   optimize=True)
        os.replace(part, board_path)
        row = {
            **selected,
            "instruction_sha256": event["instruction_sha256"],
            "proposal_path": str(proposal_path.relative_to(ROOT)),
            "proposal_sha256": prescreen_row["proposal_sha256"],
            "confirmation_evidence_path": str(
                response_path.relative_to(ROOT)),
            "confirmation_evidence_sha256": tested["sha256"],
            "board_path": str(board_path.relative_to(ROOT)),
            "board_bytes": board_path.stat().st_size,
            "board_sha256": sha256_file(board_path),
            "board_pixels": list(board.CANVAS),
            "target_branch_id": geometry_row["target"]["branch_id"],
            "alternative_branch_id": geometry_row["alternative"][
                "branch_id"],
            "branch_angle_deg": angle,
            "branch_separation_at_1_75m_m": separation,
            "reveal_interval": language_row["reveal_interval"],
            "confirmation_prefix": confirmation,
            "causal_media": linked,
            "human_status": "PENDING",
            "training_label": False,
        }
        manifest_rows.append(row)
        ui_rows.append({
            "event_id": event_id,
            "scene_id": selected["scene_id"],
            "cohort": selected["cohort"],
            "board": "boards/" + board_path.name,
            "frames": [{
                "frame_id": value["frame_id"],
                "src": "causal/%s/%s.jpg" %
                       (event_id, value["frame_id"]),
                "reveal": value["reveal_interval_member"],
                "confirm": value["confirmation_frame"],
            } for value in linked],
        })
        templates.append(template_row(event_id))
        print("%03d/%03d %s" %
              (selected["review_index"], TOTAL_COUNT, event_id), flush=True)

    manifest = {
        "manifest": "RevealNav RxR self-contained human pilot 300",
        "revision": "rxr-human-pilot-300/1",
        "status": "READY_FOR_HUMAN_REVIEW",
        "review_scope": (
            "offline branch validity plus strict online causal reveal timing; "
            "no human label exists until the completed JSONL is validated"),
        "sources": {**source_sha, str(SELECTION.relative_to(ROOT)):
                    sha256_file(SELECTION)},
        "item_count": len(manifest_rows),
        "scene_count": len({row["scene_id"] for row in manifest_rows}),
        "cohort_counts": dict(Counter(row["cohort"]
                                      for row in manifest_rows)),
        "causal_media_count": sum(len(row["causal_media"])
                                  for row in manifest_rows),
        "future_frames_in_human_causal_strips": 0,
        "panoramas_in_human_causal_strips": 0,
        "items": manifest_rows,
        "human_labels_created": 0,
        "human_reviews_completed": 0,
        "training_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    atomic_text(TEMPLATE, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in templates))
    atomic_text(REVIEWER, HTML.replace(
        "__ITEMS__", json.dumps(ui_rows, ensure_ascii=False,
                                  separators=(",", ":"))))
    atomic_text(GUIDE, (
        "# RevealNav RxR 300 条人工审核说明\n\n"
        "打开 `RXR_HUMAN_PILOT_300_REVIEWER.html`。整个目录下载到本地后"
        "仍可直接显示图片。\n\n"
        "前 250 条是按固定 SHA-256 顺序从 525 条严格候选中选出的质量审核核心；"
        "后 50 条是场景覆盖补充。论文中估计自动标注精度时只单独报告前 250 条，"
        "不能把后 50 条伪装成无偏样本。\n\n"
        "每条检查五项：\n\n"
        "1. 绿色目标和紫色备选是两条真正不同且可通行的出口；\n"
        "2. 备选不是来路、关闭门、同一出口或短假分叉；\n"
        "3. 完整指令能唯一选择绿色目标；\n"
        "4. Q 位于共享决策区，A→Q→D 时序合理；\n"
        "5. 下方严格 63° 前向历史在黄色 K=3 确认帧时已经足够，"
        "不依赖任何未来画面。\n\n"
        "五项均为是选择 ACCEPT；任一明确为否选择对应 REJECT；证据不足选择 "
        "AMBIGUOUS。网页会自动保存到浏览器 localStorage，请定期导出 "
        "`daiyang_rxr300.jsonl`。\n"))
    print(json.dumps({
        "status": manifest["status"],
        "items": manifest["item_count"],
        "scenes": manifest["scene_count"],
        "causal_media": manifest["causal_media_count"],
        "output": str(OUT_DIR.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
