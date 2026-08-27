#!/usr/bin/env python3
"""Build a compact human confirmation UI for the 16 queue50 machine rejects."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
PRIMARY = BASE / "multiview_primary"
REVIEW = BASE / "human_review_fast"
INPUTS = PRIMARY / "CR5_QUEUE50_PRIMARY_MULTIVIEW_INPUTS.json"
ACCEPTED = PRIMARY / "CR5_QUEUE50_PRIMARY_MULTIVIEW_ACCEPTED_RUN.json"
GEOMETRY = PRIMARY / "CR5_QUEUE50_DIRECTED_GEOMETRY.json"
CONTROLLER = PRIMARY / "CR5_QUEUE50_CONTROLLER_EXECUTION.json"
LEDGER = REVIEW / "CR5_QUEUE50_AUTO_REJECTED.json"
OUT_HTML = REVIEW / "CR5_QUEUE50_AUTO_REJECT_REVIEWER.html"
OUT_TEMPLATE = REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_TEMPLATE.jsonl"
OUT_MANIFEST = REVIEW / "CR5_QUEUE50_AUTO_REJECT_REVIEW_PACKAGE.json"
OUT_GUIDE = REVIEW / "机器拒绝确认说明.md"
MEDIA_DIR = REVIEW / "auto_reject_media"

EXPECTED = {
    INPUTS: "6b70a70e5eb1e25f9522b30209eb56dc2efbf6457377a1aabefdeca6886aee72",
    ACCEPTED: "0f5b643612ad1a52b12aaa12d3d26b06b5dc7b288cfbc4f435f98fd3c5b81ead",
    GEOMETRY: "46609126537ddb9c4936bc93d683dd06243b91203d6bf2f7c03f30cca7deb850",
    CONTROLLER: "567039afac8f53141b9f1d2114ee79a47611ca7e68b1eefc9d2ea40d72eff574",
    LEDGER: "14f549c8d0c73628335fa673b433593f7152fb6b6dd8a0abd074134b7c218403",
}

REASON_ZH = {
    "TARGET_DIRECTION_MISMATCH":
        "目标出口方向与参考路径不一致，超过冻结方向误差上限。",
    "TARGET_VERTICAL_MOTION_MISMATCH":
        "目标出口的上楼/下楼/同层判断与参考路径的真实高度变化不一致。",
    "TARGET_REFERENCE_ROUTE_SHORTER_THAN_1_75M":
        "目标分支在参考路径上不足 1.75 米，无法建立冻结长度的分支目标。",
    "NO_DISTINCT_EXECUTABLE_ALTERNATIVE":
        "在导航网格中没有找到与目标明显分离且可执行的第二出口。",
    "TARGET_PLANNER_ERROR":
        "冻结离散控制器两次都无法规划到目标出口，目标分支不可复现执行。",
    "ALTERNATIVE_ROLLOUT_FAIL":
        "备选出口的冻结离散控制器两次回放失败，不能作为可执行备选。",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_text(path: Path, text: str):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def replay_summary(branch):
    if branch is None:
        return None
    replays = branch.get("replays", [])
    return {
        "branch_id": branch.get("branch_id"),
        "pass": branch.get("pass"),
        "replay_statuses": [row.get("status") for row in replays],
        "final_distance_m": [row.get("final_distance_m") for row in replays],
        "collision_count": [row.get("collision_count") for row in replays],
        "deterministic_exact": branch.get("deterministic_exact"),
    }


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned review source drift: " + str(path))

    inputs = {row["event_id"]: row for row in load(INPUTS)["events"]}
    accepted = {row["event_id"]: row for row in load(ACCEPTED)["events"]}
    geometry_doc = load(GEOMETRY)
    geometry = {row["event_id"]: row for row in geometry_doc["events"]}
    controller_doc = load(CONTROLLER)
    controller = {row["event_id"]: row
                  for row in controller_doc["events"]}
    ledger = load(LEDGER)
    if (ledger.get("status") != "MACHINE_REJECTIONS_NOT_HUMAN_LABELS"
            or ledger.get("rejected_count") != 16
            or len(ledger.get("events", [])) != 16):
        raise SystemExit("machine-rejection ledger contract drift")

    items = []
    template = []
    media_manifest = []
    proposal_manifest = []
    for rejected in sorted(ledger["events"], key=lambda row: row["queue_order"]):
        event_id = rejected["event_id"]
        event = inputs[event_id]
        geometry_row = geometry[event_id]
        accepted_row = accepted[event_id]
        proposal_path = ROOT / accepted_row["accepted_proposal_path"]
        if (not proposal_path.is_file() or proposal_path.is_symlink()
                or sha256_file(proposal_path)
                != accepted_row["accepted_proposal_sha256"]):
            raise SystemExit("accepted proposal drift: " + event_id)
        proposal_manifest.append({
            "path": accepted_row["accepted_proposal_path"],
            "sha256": accepted_row["accepted_proposal_sha256"],
        })

        images = {}
        for role in ("A", "Q", "D"):
            record = event["positions"][role]["contact_sheet"]
            path = ROOT / record["path"]
            if (not path.is_file() or path.is_symlink()
                    or path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]):
                raise SystemExit("review panorama drift: " + event_id + "/" + role)
            copied = MEDIA_DIR / event_id / (role + "_PANORAMA.jpg")
            copied.parent.mkdir(parents=True, exist_ok=True)
            if copied.exists():
                if (not copied.is_file() or copied.is_symlink()
                        or copied.stat().st_size != record["bytes"]
                        or sha256_file(copied) != record["sha256"]):
                    raise SystemExit("drifted packaged panorama: " + str(copied))
            else:
                temporary = copied.with_name(copied.name + ".part")
                shutil.copyfile(path, temporary)
                os.replace(temporary, copied)
            images[role] = os.path.relpath(copied, REVIEW).replace(os.sep, "/")
            media_manifest.append({
                "event_id": event_id,
                "role": role,
                "source_path": record["path"],
                "packaged_path": str(copied.relative_to(ROOT)),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            })

        target = geometry_row.get("target") or {}
        alternative = geometry_row.get("alternative") or {}
        controller_row = controller.get(event_id)
        machine_details = {
            "geometry_status": geometry_row["status"],
            "target_branch_id": target.get("branch_id"),
            "target_descriptor": target.get("visual_descriptor"),
            "target_direction_error_deg": target.get("direction_error_deg"),
            "target_vertical_delta_m": target.get("vertical_delta_m"),
            "target_reference_future_length_m": target.get(
                "reference_future_length_available_m"),
            "alternative_branch_id": alternative.get("branch_id"),
            "alternative_descriptor": alternative.get("visual_descriptor"),
            "alternative_search_count": len(
                geometry_row.get("alternative_search", [])),
            "target_controller": replay_summary(
                controller_row.get("target") if controller_row else None),
            "alternative_controller": replay_summary(
                controller_row.get("alternative") if controller_row else None),
        }
        items.append({
            "queue_order": rejected["queue_order"],
            "event_id": event_id,
            "episode_id": rejected["episode_id"],
            "scene_id": rejected["scene_id"],
            "instruction_text": event["instruction_text"],
            "instruction_sha256": event["instruction_sha256"],
            "candidate_interval": event["candidate_interval"],
            "machine_reject_stage": rejected["automatic_reject_stage"],
            "machine_reject_reasons": rejected["automatic_reject_reasons"],
            "reason_explanations_zh": [REASON_ZH[value]
                                       for value in rejected[
                                           "automatic_reject_reasons"]],
            "images": images,
            "machine_details": machine_details,
            "human_status": "PENDING",
            "training_label": False,
        })
        template.append({
            "reviewer_id": None,
            "reviewer_type": "HUMAN",
            "event_id": event_id,
            "machine_reject_confirmed": None,
            "final_label": None,
            "reason_codes": [],
            "comment_zh": "",
        })

    if len(items) != 16 or len({row["event_id"] for row in items}) != 16:
        raise SystemExit("review item set is not exactly 16 unique events")

    items_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>CR5 机器拒绝人工确认</title>
<style>
body{margin:0;background:#101419;color:#eef2f6;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:#1c252e;padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,input{font-size:16px;padding:8px 12px}button{cursor:pointer}.confirm{background:#197648;color:white}.override{background:#ad3b3b;color:white}.amb{background:#94670f;color:white}
main{display:grid;grid-template-columns:minmax(650px,2fr) minmax(360px,1fr);gap:14px;padding:14px}.views{display:grid;gap:12px}.view{background:#171d23;padding:8px;border:1px solid #46515d}.view b{display:block;margin-bottom:6px}.view img{display:block;width:100%;height:auto}.info{background:#171d23;padding:16px;border:1px solid #46515d;position:sticky;top:78px;align-self:start;max-height:calc(100vh - 110px);overflow:auto}.reason{background:#342b16;border-left:5px solid #e2a72f;padding:10px;margin:8px 0}.instruction{white-space:pre-wrap;background:#202933;padding:12px;line-height:1.45}.details{font-family:ui-monospace,monospace;white-space:pre-wrap;font-size:13px;color:#c8d2dc}.controls{display:grid;gap:8px;margin-top:14px}.muted{color:#aeb8c2;font-size:14px}.status{padding:4px 8px;border-radius:4px;background:#2d3742}
@media(max-width:1000px){main{grid-template-columns:1fr}.info{position:static;max-height:none}}
</style></head><body>
<header><b id="progress"></b><span id="event" class="status"></span>
<label>审核者 <input id="reviewer" value="daiyang" size="12"></label>
<button onclick="move(-1)">← 上一条</button><button onclick="move(1)">下一条 →</button>
<button onclick="exportJsonl()">导出 JSONL</button>
<label><button onclick="document.getElementById('importer').click()">导入进度</button><input id="importer" type="file" hidden></label></header>
<main><section class="views">
<div class="view"><b>A：到达候选点之前</b><img id="imgA"></div>
<div class="view"><b>Q：机器判断的候选决策位置</b><img id="imgQ"></div>
<div class="view"><b>D：沿参考路线继续之后</b><img id="imgD"></div>
</section><aside class="info">
<h2 id="title"></h2><div id="reasons"></div>
<h3>完整指令</h3><div id="instruction" class="instruction"></div>
<h3>机器证据摘要</h3><div id="details" class="details"></div>
<div class="controls">
<button class="confirm" onclick="label('CONFIRM_REJECT')">C：确认机器拒绝合理</button>
<button class="override" onclick="label('SUSPECT_FALSE_REJECT')">O：怀疑误拒，需要重跑/完整复核</button>
<button class="amb" onclick="label('AMBIGUOUS')">U：图像或证据不足</button>
<input id="comment" placeholder="可选中文备注"><button onclick="clearCurrent()">清除本条</button>
</div><p class="muted">通过标准：只要列出的任一硬失败真实成立，就点 C。只有你认为失败原因与图像/证据明显矛盾时才点 O；无法判断点 U。快捷键 C/O/U、左右方向键。每次选择自动保存并进入下一条。</p>
</aside></main>
<script>
const items=__ITEMS__;
let index=0;let labels=JSON.parse(localStorage.getItem('cr5_auto_reject16_labels_v1')||'{}');
function baseRow(kind){return {reviewer_id:document.getElementById('reviewer').value||null,reviewer_type:'HUMAN',event_id:items[index].event_id,machine_reject_confirmed:kind==='CONFIRM_REJECT'?true:kind==='SUSPECT_FALSE_REJECT'?false:null,final_label:kind,reason_codes:kind==='CONFIRM_REJECT'?items[index].machine_reject_reasons:kind==='SUSPECT_FALSE_REJECT'?['SUSPECT_FALSE_REJECT']:['INSUFFICIENT_EVIDENCE'],comment_zh:document.getElementById('comment').value||''}}
function label(kind){let row=baseRow(kind);labels[row.event_id]=row;localStorage.setItem('cr5_auto_reject16_labels_v1',JSON.stringify(labels));render();setTimeout(()=>move(1),160)}
function clearCurrent(){delete labels[items[index].event_id];localStorage.setItem('cr5_auto_reject16_labels_v1',JSON.stringify(labels));render()}
function move(delta){index=Math.max(0,Math.min(items.length-1,index+delta));render()}
function render(){let it=items[index],r=labels[it.event_id];document.getElementById('progress').textContent=(index+1)+' / '+items.length+'（已完成 '+Object.keys(labels).length+'）';document.getElementById('event').textContent=it.event_id+(r?' | '+r.final_label:' | 未标');document.getElementById('title').textContent='队列 '+String(it.queue_order).padStart(2,'0')+' · '+it.machine_reject_stage;document.getElementById('imgA').src=it.images.A;document.getElementById('imgQ').src=it.images.Q;document.getElementById('imgD').src=it.images.D;document.getElementById('instruction').textContent=it.instruction_text;document.getElementById('reasons').innerHTML=it.machine_reject_reasons.map((x,i)=>'<div class="reason"><b>'+x+'</b><br>'+it.reason_explanations_zh[i]+'</div>').join('');document.getElementById('details').textContent=JSON.stringify(it.machine_details,null,2);document.getElementById('comment').value=r?.comment_zh||''}
function exportJsonl(){let rows=items.filter(x=>labels[x.event_id]).map(x=>{let r=labels[x.event_id];r.reviewer_id=document.getElementById('reviewer').value||r.reviewer_id;return JSON.stringify(r)});let blob=new Blob([rows.join('\\n')+(rows.length?'\\n':'')],{type:'application/jsonl'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='daiyang_auto_reject16.jsonl';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importer').addEventListener('change',async e=>{let text=await e.target.files[0].text();for(let line of text.split(/\\r?\\n/)){if(!line.trim())continue;let row=JSON.parse(line);if(items.some(x=>x.event_id===row.event_id))labels[row.event_id]=row}localStorage.setItem('cr5_auto_reject16_labels_v1',JSON.stringify(labels));render()});
document.addEventListener('keydown',e=>{if(document.activeElement.id==='comment'||document.activeElement.id==='reviewer')return;let k=e.key.toLowerCase();if(k==='c')label('CONFIRM_REJECT');else if(k==='o')label('SUSPECT_FALSE_REJECT');else if(k==='u')label('AMBIGUOUS');else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});render();
</script></body></html>""".replace("__ITEMS__", items_json)
    atomic_text(OUT_HTML, page)
    atomic_text(OUT_TEMPLATE, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in template))

    manifest = {
        "manifest": "MF2-CR5 queue50 machine-reject human confirmation",
        "revision": "cr5-queue50-auto-reject-human-review/1",
        "status": "READY_FOR_16_ITEM_HUMAN_CONFIRMATION",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "item_count": len(items),
        "items": items,
        "media_manifest": media_manifest,
        "proposal_manifest": proposal_manifest,
        "reviewer_html": {
            "path": str(OUT_HTML.relative_to(ROOT)),
            "bytes": OUT_HTML.stat().st_size,
            "sha256": sha256_file(OUT_HTML),
        },
        "template": {
            "path": str(OUT_TEMPLATE.relative_to(ROOT)),
            "bytes": OUT_TEMPLATE.stat().st_size,
            "sha256": sha256_file(OUT_TEMPLATE),
        },
        "human_labels_created": 0,
        "training_authorized": False,
    }
    atomic_text(OUT_MANIFEST, json.dumps(
        manifest, indent=2, ensure_ascii=False) + "\n")
    atomic_text(OUT_GUIDE, """# 16 条机器拒绝的人工确认

打开 `CR5_QUEUE50_AUTO_REJECT_REVIEWER.html`。每条只需判断机器拒绝是否合理：

- 按 `C`：确认至少一个硬失败成立，这条应保持拒绝；
- 按 `O`：失败原因与图像或证据明显矛盾，可能是误拒；
- 按 `U`：目前证据不足，不能确认。

图中 A 是候选点之前，Q 是机器候选决策位置，D 是继续沿参考路线之后。这里不是重新寻找分支，也不要求给出目标/备选标签。三张图用于发现明显的语义错误；导航网格与控制器数值列在右侧。

页面所需的 48 张图片均在本目录的 `auto_reject_media/` 中；下载整个 `human_review_fast` 文件夹即可离线使用，不依赖相邻目录。

完成 16/16 后点击“导出 JSONL”，文件名为 `daiyang_auto_reject16.jsonl`。请把它放回本目录或上传给主 agent。只有 16 条全部获得人工结论后，冻结的 50/50 人工复核门才可重新评估。
""")
    print(json.dumps({
        "status": manifest["status"],
        "items": len(items),
        "html": str(OUT_HTML.relative_to(ROOT)),
        "manifest": str(OUT_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256_file(OUT_MANIFEST),
        "template": str(OUT_TEMPLATE.relative_to(ROOT)),
        "guide": str(OUT_GUIDE.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
