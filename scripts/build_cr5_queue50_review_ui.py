#!/usr/bin/env python3
"""Build a dependency-free keyboard review UI for the queue50 packet."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50/human_review_fast"
MANIFEST = BASE / "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json"
OUT = BASE / "CR5_QUEUE50_REVIEWER.html"


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>CR5 50条扩展快速审核</title>
<style>
body{margin:0;background:#11151a;color:#edf1f5;font-family:system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:#1b222a;padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button,input{font-size:16px;padding:8px 12px}.ok{background:#16854c;color:white}.bad{background:#a53535;color:white}.amb{background:#9a6a10;color:white}
#board{display:block;max-width:98vw;max-height:78vh;margin:12px auto;border:1px solid #4b5561}
#controls{padding:8px 16px 22px;display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:8px}
#comment{grid-column:1/-1;width:calc(100% - 26px)} .wide{grid-column:1/-1}.muted{color:#aeb8c2}
</style></head><body>
<header><b id="progress"></b><span id="event"></span>
<label>审核者 <input id="reviewer" value="daiyang" size="12"></label>
<button onclick="move(-1)">← 上一条</button><button onclick="move(1)">下一条 →</button>
<button onclick="exportJsonl()">导出 JSONL</button><label><button onclick="document.getElementById('importer').click()">导入进度</button><input id="importer" type="file" hidden></label>
</header>
<img id="board" alt="review board">
<div id="controls">
<button class="ok wide" onclick="accept()">A：四项都通过 → ACCEPT</button>
<button class="bad" onclick="reject(0)">1：没有两条不同可走出口</button>
<button class="bad" onclick="reject(1)">2：备选是来路/关闭/重复/假分叉</button>
<button class="bad" onclick="reject(2)">3：指令不能唯一选中绿色目标</button>
<button class="bad" onclick="reject(3)">4：Q位置或A→Q→D时序不合理</button>
<button class="amb" onclick="ambiguous()">U：证据不足 → AMBIGUOUS</button>
<button onclick="clearCurrent()">清除本条</button>
<input id="comment" placeholder="可选中文备注；REJECT时可补充具体原因">
<div class="wide muted">快捷键：A 接受；1/2/3/4 按对应原因拒绝；U 模糊；←/→ 翻页。每次选择自动保存并跳到下一条。机器预筛不是人工标签。</div>
</div>
<script>
const items=__ITEMS__;
const keys=['two_distinct_executable_exits','alternative_is_not_incoming_closed_or_duplicate','instruction_uniquely_selects_target','decision_center_and_temporal_order_are_reasonable'];
const reasonCodes=['NO_TWO_DISTINCT_EXECUTABLE_EXITS','ALTERNATIVE_INCOMING_CLOSED_DUPLICATE_OR_SHORT','INSTRUCTION_TARGET_NOT_UNIQUE','DECISION_CENTER_OR_TEMPORAL_ORDER_INVALID'];
let index=0;let labels=JSON.parse(localStorage.getItem('cr5_queue50_labels_v1')||'{}');
function baseRow(){return {reviewer_id:document.getElementById('reviewer').value||null,reviewer_type:'HUMAN',event_id:items[index].event_id,two_distinct_executable_exits:null,alternative_is_not_incoming_closed_or_duplicate:null,instruction_uniquely_selects_target:null,decision_center_and_temporal_order_are_reasonable:null,final_label:null,reason_codes:[],comment_zh:document.getElementById('comment').value||''}}
function save(row){labels[row.event_id]=row;localStorage.setItem('cr5_queue50_labels_v1',JSON.stringify(labels));render();setTimeout(()=>move(1),180)}
function accept(){let r=baseRow();keys.forEach(k=>r[k]=true);r.final_label='ACCEPT';save(r)}
function reject(i){let r=baseRow();r[keys[i]]=false;r.final_label='REJECT';r.reason_codes=[reasonCodes[i]];save(r)}
function ambiguous(){let r=baseRow();r.final_label='AMBIGUOUS';r.reason_codes=['INSUFFICIENT_VISUAL_EVIDENCE'];save(r)}
function clearCurrent(){delete labels[items[index].event_id];localStorage.setItem('cr5_queue50_labels_v1',JSON.stringify(labels));render()}
function move(delta){index=Math.max(0,Math.min(items.length-1,index+delta));render()}
function render(){let it=items[index],r=labels[it.event_id];document.getElementById('board').src=it.board;document.getElementById('event').textContent=it.event_id+' | '+it.priority+(r?' | 已标 '+r.final_label:' | 未标');document.getElementById('progress').textContent=(index+1)+' / '+items.length+'（已完成 '+Object.keys(labels).length+'）';document.getElementById('comment').value=r?.comment_zh||''}
function exportJsonl(){let rows=items.filter(x=>labels[x.event_id]).map(x=>{let r=labels[x.event_id];r.reviewer_id=document.getElementById('reviewer').value||r.reviewer_id;return JSON.stringify(r)});let blob=new Blob([rows.join('\n')+(rows.length?'\n':'')],{type:'application/jsonl'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='daiyang_queue50.jsonl';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importer').addEventListener('change',async e=>{let text=await e.target.files[0].text();for(let line of text.split(/\r?\n/)){if(!line.trim())continue;let r=JSON.parse(line);if(items.some(x=>x.event_id===r.event_id))labels[r.event_id]=r}localStorage.setItem('cr5_queue50_labels_v1',JSON.stringify(labels));render()});
document.addEventListener('keydown',e=>{if(document.activeElement.id==='comment'||document.activeElement.id==='reviewer')return;if(e.key.toLowerCase()==='a')accept();else if(e.key.toLowerCase()==='u')ambiguous();else if(['1','2','3','4'].includes(e.key))reject(Number(e.key)-1);else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});render();
</script></body></html>'''


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    items = [{
        "event_id": row["event_id"],
        "board": "boards/" + Path(row["board_path"]).name,
        "priority": row["automatic_review_priority"],
    } for row in manifest["items"]]
    if len(items) != 34:
        raise SystemExit("review manifest must contain 34 items")
    OUT.write_text(HTML.replace(
        "__ITEMS__", json.dumps(items, ensure_ascii=False,
                                  separators=(",", ":"))), encoding="utf-8")
    print(json.dumps({"status": "PASS", "items": len(items),
                      "output": str(OUT.relative_to(ROOT))},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
