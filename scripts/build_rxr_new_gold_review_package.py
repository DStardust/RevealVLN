#!/usr/bin/env python3
"""Build a self-contained 900-candidate, three-lane new-Gold review package."""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v1"
SELECTION = BASE / "RXR_SCALE_V1_SELECTION.json"
INPUTS = BASE / "new_gold/multiview/RXR_SCALE_MULTIVIEW_INPUTS.json"
OUT = BASE / "new_gold/review_package"
MEDIA = OUT / "media"
MANIFEST = OUT / "RXR_NEW_GOLD_REVIEW_MANIFEST.json"
GUIDE = OUT / "审核说明.md"
TARGET = 900
LANES = ("R1", "R2", "R3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(value, encoding="utf-8")
    os.replace(part, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def link(source: Path, destination: Path, expected: dict) -> dict:
    if not (
        source.is_file()
        and not source.is_symlink()
        and ROOT.resolve() in source.resolve().parents
        and source.stat().st_size == expected["bytes"]
        and sha256_file(source) == expected["sha256"]
    ):
        raise RuntimeError("review media source drift: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not (
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == expected["bytes"]
            and sha256_file(destination) == expected["sha256"]
        ):
            raise RuntimeError("existing review media drift: " + str(destination))
    else:
        part = destination.with_name(destination.name + ".part")
        os.link(source, part)
        os.replace(part, destination)
    return {
        "path": str(destination.relative_to(OUT)),
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
        "pixels": expected["pixels"],
    }


def blank(event_id: str, lane: str) -> dict:
    return {
        "schema_version": "revealnav-new-gold-human-review/1",
        "review_lane": lane,
        "reviewer_id": None,
        "reviewer_type": "HUMAN",
        "event_id": event_id,
        "event_valid": None,
        "q_state": None,
        "target_in_set_at_q": None,
        "candidate_separable_at_q": None,
        "decisive_evidence_closed_at_q": None,
        "multiple_executable_branches": None,
        "reason_codes": [],
        "comment": "",
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>RevealNav 新三审 Gold</title><style>
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:#101419;color:#edf2f7;font-family:system-ui,sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{flex:none;z-index:3;background:#1c252e;padding:8px 12px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,input{font-size:14px;padding:7px}.workspace{flex:1;min-height:0;display:grid;grid-template-columns:minmax(620px,64fr) minmax(430px,36fr);gap:8px;padding:8px}
.visual{min-width:0;min-height:0;display:flex;flex-direction:column;background:#151c23;border:1px solid #354454}.media-tabs{flex:none;display:flex;gap:6px;padding:7px;overflow-x:auto;background:#1c252e}.media-tab{min-width:70px;background:#293541;color:#edf2f7;border:2px solid #607080}.media-tab.selected{background:#17623f;border-color:#73d69f}.stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:6px;overflow:hidden}.stage img{display:block;max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.media-hint{flex:none;padding:5px 9px;color:#aeb8c2;text-align:center}
.side{min-height:0;display:grid;grid-template-rows:minmax(150px,28vh) minmax(0,1fr);gap:8px}.instruction{min-height:0;background:#202a34;border:1px solid #485663;padding:12px;overflow:auto;font-size:16px;line-height:1.48;white-space:pre-wrap}.controls{min-height:0;overflow-y:auto;padding:8px;background:#151c23;border:1px solid #354454}.question{margin-bottom:7px;padding:8px;background:#182029;border:1px solid #354454}.question b{display:block;margin-bottom:6px;font-size:15px}.choices{display:flex;gap:7px;flex-wrap:wrap}.choice{min-width:96px;border:2px solid #607080;background:#293541;color:#edf2f7}.choice.selected{border-color:#73d69f;background:#17623f}.derived{margin:7px 0;padding:8px;background:#202a34;border-left:4px solid #73d69f;font-size:15px}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:7px}.wide{width:100%;margin-top:6px}.ok{background:#18764a;color:white}.bad{background:#963f3f;color:white}.amb{background:#946813;color:white}.muted{color:#aeb8c2;font-size:13px}.hidden{display:none}
.zoom{display:none;position:fixed;inset:0;z-index:10;background:rgba(0,0,0,.94);align-items:center;justify-content:center;padding:12px}.zoom.open{display:flex}.zoom img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-out}
@media(max-width:1050px){body{overflow:auto;display:block}.workspace{height:auto;grid-template-columns:1fr}.visual{height:75vh}.side{grid-template-rows:minmax(180px,35vh) auto}.controls{overflow:visible}}
</style></head><body><header><b id="progress"></b><span id="event"></span>
<label>审核者 <input id="reviewer" placeholder="真实姓名或固定ID" size="18"></label>
<button onclick="move(-1)">← 上一条</button><button onclick="move(1)">下一条 →</button><button onclick="exportRows()">导出本审核者 JSONL</button>
<label><button onclick="document.getElementById('importer').click()">导入进度</button><input id="importer" type="file" hidden></label></header>
<div class="workspace"><main class="visual"><div id="media_tabs" class="media-tabs"></div><div class="stage"><img id="active_media" onclick="openZoom()"></div><div class="media-hint">默认显示 Q；点击 A/Q/D/上下文切换，点击大图可全屏放大（快捷键 A/Q/D）</div></main><aside class="side"><div id="instruction" class="instruction"></div>
<div class="controls">
<div class="question"><b>① 这是有效的真实决策分叉吗？</b><div class="choices">
<button class="choice" data-field="event_valid" data-value="ACCEPT" onclick="choose('event_valid','ACCEPT')">是，继续判断</button>
<button class="bad" onclick="quickReject()">否：保存并下一条</button><button class="amb" onclick="quickAmbiguous()">看不清：保存并下一条</button></div></div>
<div id="semantics" class="hidden">
<div class="question"><b>② 在 Q 全景中，正确出口/目标分支已经出现了吗？</b><div id="target_in_set_at_q" class="choices"></div></div>
<div id="separable_question" class="question"><b>③ 在 Q 时，能把正确出口和其他出口区分开吗？</b><div id="candidate_separable_at_q" class="choices"></div></div>
<div class="question"><b>④ 到 Q 时，指令中决定选哪个出口的语言证据已经足够了吗？</b><div id="decisive_evidence_closed_at_q" class="choices"></div></div>
<label class="question"><input id="unresolvable" type="checkbox" onchange="updateDerived()"> 少见情况：直到最后安全转向点仍无法判断正确分支（UNRESOLVABLE）</label>
</div>
<div id="derived" class="derived">尚未标注</div>
<input id="comment" class="wide" placeholder="可选备注">
<div class="actions"><button class="ok" onclick="saveAndNext()">保存并下一条</button><button class="amb" onclick="quickAmbiguous()">任一关键项不确定</button></div>
<div class="wide muted">底层六字段仍完整保存：事件有效性和多分支由第①项共同写入，U/A/D 按第②～④项自动推导。目标未出现时“候选可区分”自动记为 NO。</div></div></aside></div>
<div id="zoom" class="zoom" onclick="closeZoom()"><img id="zoom_media"></div>
<script>const lane=__LANE__;const items=__ITEMS__;const storage='revealnav_new_gold_'+lane;let labels=JSON.parse(localStorage.getItem(storage)||'{}');let index=0;let draft={};let activeMedia=1;
const semanticFields=['target_in_set_at_q','candidate_separable_at_q','decisive_evidence_closed_at_q'];
function semanticButtons(field){let box=document.getElementById(field);box.innerHTML='';for(let [value,text] of [['YES','是'],['NO','否']]){let b=document.createElement('button');b.className='choice'+(draft[field]===value?' selected':'');b.textContent=text;b.onclick=()=>choose(field,value);box.append(b)}}
function deriveQ(){if(draft.q_state==='UNRESOLVABLE')return 'UNRESOLVABLE';if(draft.target_in_set_at_q==='NO')return 'U';if(draft.target_in_set_at_q==='YES'&&draft.candidate_separable_at_q&&draft.decisive_evidence_closed_at_q)return draft.candidate_separable_at_q==='YES'&&draft.decisive_evidence_closed_at_q==='YES'?'D':'A';return null}
function choose(field,value){draft[field]=value;if(field==='event_valid'&&value==='ACCEPT')draft.multiple_executable_branches='YES';if(field==='target_in_set_at_q'&&value==='NO')draft.candidate_separable_at_q='NO';if(field==='target_in_set_at_q'&&value==='YES'&&draft.candidate_separable_at_q==='NO')draft.candidate_separable_at_q=null;draft.q_state=deriveQ();renderControls()}
function updateDerived(){draft.q_state=document.getElementById('unresolvable').checked?'UNRESOLVABLE':null;draft.q_state=deriveQ();renderControls()}
function row(){let it=items[index];return {schema_version:'revealnav-new-gold-human-review/1',review_lane:lane,reviewer_id:document.getElementById('reviewer').value||null,reviewer_type:'HUMAN',event_id:it.event_id,event_valid:draft.event_valid||null,q_state:deriveQ(),target_in_set_at_q:draft.target_in_set_at_q||null,candidate_separable_at_q:draft.candidate_separable_at_q||null,decisive_evidence_closed_at_q:draft.decisive_evidence_closed_at_q||null,multiple_executable_branches:draft.multiple_executable_branches||null,reason_codes:[],comment:document.getElementById('comment').value||''}}
function persist(){let r=row();if(r.event_valid==='ACCEPT'&&(!r.q_state||semanticFields.some(f=>!r[f]))){alert('通过事件还缺少第②～④项；看不清请点“不确定”');return false}labels[r.event_id]=r;localStorage.setItem(storage,JSON.stringify(labels));render();return true}
function saveAndNext(){if(persist())move(1)}function quickReject(){draft={event_valid:'REJECT',multiple_executable_branches:'NO'};if(persist())move(1)}function quickAmbiguous(){draft={event_valid:'AMBIGUOUS',multiple_executable_branches:'AMBIGUOUS'};if(persist())move(1)}
function move(d){index=Math.max(0,Math.min(items.length-1,index+d));render()}
function media(){let it=items[index];return [...it.panoramas.map(x=>({label:x.role,path:x.path})),...it.context.map((x,i)=>({label:'上下文 '+(i+1),path:x.path}))]}
function showMedia(i){let all=media();activeMedia=Math.max(0,Math.min(all.length-1,i));document.getElementById('active_media').src=all[activeMedia].path;let tabs=document.getElementById('media_tabs');tabs.innerHTML='';all.forEach((x,j)=>{let b=document.createElement('button');b.className='media-tab'+(j===activeMedia?' selected':'');b.textContent=x.label;b.onclick=()=>showMedia(j);tabs.append(b)})}
function openZoom(){document.getElementById('zoom_media').src=document.getElementById('active_media').src;document.getElementById('zoom').classList.add('open')}function closeZoom(){document.getElementById('zoom').classList.remove('open')}
function renderControls(){document.getElementById('semantics').classList.toggle('hidden',draft.event_valid!=='ACCEPT');for(let b of document.querySelectorAll('[data-field="event_valid"]'))b.classList.toggle('selected',draft.event_valid===b.dataset.value);for(let f of semanticFields)semanticButtons(f);document.getElementById('separable_question').classList.toggle('hidden',draft.target_in_set_at_q==='NO');document.getElementById('unresolvable').checked=draft.q_state==='UNRESOLVABLE';let q=deriveQ(),complete=draft.event_valid==='ACCEPT'&&q&&semanticFields.every(f=>draft[f]);document.getElementById('derived').textContent=draft.event_valid==='REJECT'?'将保存为 REJECT（非有效分叉）':draft.event_valid==='AMBIGUOUS'?'将保存为 AMBIGUOUS（不进入 Gold）':complete?'自动结果：ACCEPT / '+q+' / 多分支=YES':draft.event_valid==='ACCEPT'?'有效分叉；请完成第②～④项':'尚未标注'}
function render(){let it=items[index],r=labels[it.event_id]||{};draft={...r};activeMedia=1;document.getElementById('progress').textContent=(index+1)+' / '+items.length+'（已保存 '+Object.keys(labels).length+'）';document.getElementById('event').textContent=lane+' | '+it.blind_id+(r.event_valid?' | '+r.event_valid:'');document.getElementById('instruction').textContent='完整指令\n\n'+it.instruction;document.getElementById('comment').value=r.comment||'';showMedia(activeMedia);renderControls()}
function exportRows(){let reviewer=document.getElementById('reviewer').value;if(!reviewer){alert('请填写真实审核者ID');return}let rows=items.filter(x=>labels[x.event_id]).map(x=>{let r=labels[x.event_id];r.reviewer_id=reviewer;return JSON.stringify(r)});let b=new Blob([rows.join('\n')+(rows.length?'\n':'')],{type:'application/jsonl'});let a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='new_gold_'+lane+'_'+reviewer+'.jsonl';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importer').addEventListener('change',async e=>{let text=await e.target.files[0].text();for(let line of text.split(/\r?\n/)){if(!line.trim())continue;let r=JSON.parse(line);if(r.review_lane===lane&&items.some(x=>x.event_id===r.event_id))labels[r.event_id]=r}localStorage.setItem(storage,JSON.stringify(labels));render()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeZoom();else if(e.ctrlKey&&e.key.toLowerCase()==='s'){e.preventDefault();persist()}else if(document.activeElement.tagName!=='INPUT'&&e.key==='ArrowLeft')move(-1);else if(document.activeElement.tagName!=='INPUT'&&e.key==='ArrowRight')move(1);else if(document.activeElement.tagName!=='INPUT'&&['a','q','d'].includes(e.key.toLowerCase()))showMedia({a:0,q:1,d:2}[e.key.toLowerCase()])});render();</script></body></html>'''


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    inputs = json.loads(INPUTS.read_text())
    if not (
        selection.get("status") == "SCALE_V1_SELECTION_FROZEN"
        and inputs.get("status") == "READY_FOR_BRANCH_PROPOSER"
        and inputs.get("lane") == "new_gold"
        and inputs.get("old_gold_payload_read") is False
    ):
        raise RuntimeError("new Gold review precondition failed")
    input_by_id = {row["event_id"]: row for row in inputs["events"]}
    ordered = sorted(selection["new_gold"], key=lambda row: (row["gold_wave"], row["gold_rank_commitment"]))
    selected = [row for row in ordered if row["event_id"] in input_by_id][:TARGET]
    if len(selected) != TARGET:
        raise RuntimeError(f"only {len(selected)} rendered new-Gold candidates")

    manifest_items = []
    ui_by_id = {}
    for review_index, selected_row in enumerate(selected, 1):
        event = input_by_id[selected_row["event_id"]]
        event_dir = MEDIA / event["event_id"]
        panoramas = []
        for role in ("A", "Q", "D"):
            record = event["positions"][role]["contact_sheet"]
            destination = event_dir / f"{role}_PANORAMA.jpg"
            linked = link(ROOT / record["path"], destination, record)
            panoramas.append({"role": role, **linked})
        context = []
        for offset, record in enumerate(event["chronological_context_storyboards"]):
            destination = event_dir / f"CONTEXT_{offset:02d}.jpg"
            linked = link(ROOT / record["path"], destination, record)
            context.append(linked)
        blind_id = f"NG{review_index:04d}"
        item = {
            "review_index": review_index,
            "blind_id": blind_id,
            "event_id": event["event_id"],
            "episode_id": event["episode_id"],
            "scene_id": event["scene_id"],
            "instruction_sha256": event["instruction_sha256"],
            "panoramas": panoramas,
            "context": context,
            "source_gold_wave": selected_row["gold_wave"],
        }
        manifest_items.append(item)
        ui_by_id[event["event_id"]] = {
            "blind_id": blind_id,
            "event_id": event["event_id"],
            "instruction": event["instruction_text"],
            "panoramas": [{"role": row["role"], "path": row["path"]} for row in panoramas],
            "context": [{"path": row["path"]} for row in context],
        }

    lane_files = {}
    for lane in LANES:
        order = sorted(
            ui_by_id,
            key=lambda event_id: hashlib.sha256(f"new-gold-{lane}/1|{event_id}".encode()).hexdigest(),
        )
        lane_items = [ui_by_id[event_id] for event_id in order]
        template = OUT / f"RXR_NEW_GOLD_{lane}_TEMPLATE.jsonl"
        atomic_text(template, "".join(json.dumps(blank(event_id, lane), ensure_ascii=False) + "\n" for event_id in order))
        reviewer = OUT / f"RXR_NEW_GOLD_{lane}_REVIEWER.html"
        body = HTML.replace("__LANE__", json.dumps(lane)).replace(
            "__ITEMS__", json.dumps(lane_items, ensure_ascii=False).replace("</", "<\\/")
        )
        atomic_text(reviewer, body)
        lane_files[lane] = {
            "template": str(template.relative_to(ROOT)),
            "template_sha256": sha256_file(template),
            "reviewer": str(reviewer.relative_to(ROOT)),
            "reviewer_sha256": sha256_file(reviewer),
            "order_commitment_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
        }

    guide = """# RevealNav 新 Gold 三人独立审核（快速版）\n\n每位审核者只打开分配给自己的 R1、R2 或 R3 HTML，不查看他人结果。\n整个 `review_package` 目录必须一起下载，图片才能离线显示。\n\n## 每条怎么标\n\n1. 先判断它是不是有效的真实决策分叉：至少有两个可执行出口，并且图片、指令和路线时序没有明显错误。\n2. 如果不是，点“否”；如果任何关键信息看不清，点“看不清”。这两种情况都会立即保存并进入下一条。\n3. 只有选择“是”时，再判断 Q 时正确出口是否出现、是否能与其他出口区分、指令是否已经足以决定选哪个出口。\n4. 页面会自动推导并保存完整的事件有效性、多分支以及 U/A/D 字段，不需要人工重复填写。\n5. 仅当正确分支直到最后安全转向点仍无法判断时，勾选 UNRESOLVABLE。不要把普通 U 或 A 勾成不可解。\n\n不确定时不要猜。定期导出 JSONL。模板为空不代表已经完成审核。只有三份由不同真实审核者导出的完整 JSONL\n通过聚合器后，事件才可能进入新 Gold。MLLM 结果只能作为预筛，不能冒充 HUMAN。\n"""
    atomic_text(GUIDE, guide)
    manifest = {
        "schema_version": "revealnav-new-gold-review-package/1",
        "status": "PENDING_THREE_INDEPENDENT_HUMAN_REVIEWS",
        "sources": {
            str(SELECTION.relative_to(ROOT)): sha256_file(SELECTION),
            str(INPUTS.relative_to(ROOT)): sha256_file(INPUTS),
        },
        "selection": {
            "review_candidates": TARGET,
            "minimum_final_gold": 600,
            "wave1_then_precommitted_reserve": True,
            "selection_used_human_labels": False,
            "prior_event_ids_excluded": True,
        },
        "review_lanes": lane_files,
        "review_interface": "quick-derived-six-field/2",
        "items": manifest_items,
        "media_files": sum(len(row["panoramas"]) + len(row["context"]) for row in manifest_items),
        "logical_media_bytes": sum(
            media["bytes"]
            for row in manifest_items
            for media in row["panoramas"] + row["context"]
        ),
        "human_labels_created": 0,
        "three_reviewer_agreement_measured": False,
        "old_gold_payload_read": False,
        "gold_authorized": False,
        "paper_result": False,
    }
    atomic_json(MANIFEST, manifest)
    print(json.dumps({
        "status": manifest["status"], "items": len(manifest_items),
        "media_files": manifest["media_files"],
        "output": str(MANIFEST.relative_to(ROOT)),
        "sha256": sha256_file(MANIFEST),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
