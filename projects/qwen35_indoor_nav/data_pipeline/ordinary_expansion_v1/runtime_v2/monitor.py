"""Read-only loopback data-expansion dashboard. Does not import the GPU stack."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'envdrop_production_v2'

def read(path):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return {}

def collect():
    lanes=[]
    for gpu in (3,4,5):
        assigned=list(range(gpu-3,42,3));states=[]
        for shard in assigned:
            root=DATA/f'shard_{shard:04d}'
            p=read(root/'PROGRESS.json')
            if p:states.append(dict(shard=shard,**p))
        out=HERE/f'lanes/gpu_{gpu}/attempt_000'
        final=read(out/'RESULT.json');gate=read(out/'SENTINEL_GATE.json')
        count=lambda key:sum(p.get(key,0) for p in states)
        lanes.append(dict(gpu=gpu,target=3155 if gpu==5 else 3253,
            completed=count('completed'),replay_certified=count('replay_certified_routes'),
            strict_routes=count('audited_routes'),strict_decisions=count('audited_instruction_conditioned_decisions'),
            sentinel_pass=gate.get('strict_pass'),sentinel=gate,
            stage=('FAILED' if final.get('error') else 'CLOSED') if final else ('PRODUCING' if gate.get('strict_pass') else 'FIRST_STRICT_GATE'),
            latest_shard=states[-1]['shard'] if states else None,result=final))
    return dict(time_unix=time.time(),read_only=True,training_status='STOPPED',checkpoint_updates=51301,
        total_routes=9661,lanes=lanes,completed=sum(l['completed'] for l in lanes),
        strict_routes=sum(l['strict_routes'] for l in lanes),strict_decisions=sum(l['strict_decisions'] for l in lanes),
        old_data_audit=read(HERE.parent/'recovery_inventory_20260911_v1/run_001/PROGRESS.json'),
        final_merge=read(HERE/'merge/RESULT.json'),launch=read(HERE/'launch_001/RESULT.json'),
        note='逐片严格合格量不等于最终合并入训；旧数据恢复单列。本波固定未尝试路线，不自动重启训练。')

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>普通导航数据扩产</title>
<style>body{font:16px/1.6 system-ui;background:#101b2c;color:#e4ecf8;max-width:1000px;margin:35px auto;padding:0 20px}progress{width:100%;height:24px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #456;text-align:left}pre{white-space:pre-wrap}a{color:#9bd}#count{font-size:24px}</style>
<h1>普通导航数据扩产 · 三卡 V2</h1><p>训练已停止，最终检查点 51,301 步。GPU1 完整基准评测独立运行。</p><p id="count">读取中</p><progress id="bar" max="9661" value="0"></progress><p id="note"></p>
<table><thead><tr><th>GPU</th><th>状态</th><th>已处理 / 计划</th><th>严格合格路线</th><th>合格动作</th></tr></thead><tbody id="rows"></tbody></table><h2>旧数据 CPU 恢复审核（单列）</h2><pre id="old"></pre><h2>最终合并</h2><pre id="final"></pre><p>只读，10 秒刷新 · <a href="/api/status">JSON 接口</a></p>
<script>async function refresh(){try{let d=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('count').textContent=`新波已处理 ${d.completed} / ${d.total_routes} 条；严格合格 ${d.strict_routes} 条 / ${d.strict_decisions} 动作`;document.getElementById('bar').value=d.completed;document.getElementById('note').textContent=d.note;let rows=document.getElementById('rows');rows.replaceChildren();for(let x of d.lanes){let tr=document.createElement('tr');for(let value of [x.gpu,x.stage,`${x.completed} / ${x.target}`,x.strict_routes,x.strict_decisions]){let td=document.createElement('td');td.textContent=value;tr.appendChild(td);}rows.appendChild(tr);}document.getElementById('old').textContent=JSON.stringify(d.old_data_audit,null,2);document.getElementById('final').textContent=Object.keys(d.final_merge).length?JSON.stringify(d.final_merge,null,2):'尚未全批闭合、复核合并。';}catch(e){document.getElementById('count').textContent='状态读取失败：'+e;}}refresh();setInterval(refresh,10000);</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/':body=PAGE.encode();kind='text/html; charset=utf-8'
        elif self.path=='/api/status':body=json.dumps(collect(),ensure_ascii=False,allow_nan=False).encode();kind='application/json; charset=utf-8'
        elif self.path=='/healthz':body=b'{"ok":true,"read_only":true}';kind='application/json'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_POST(self):self.send_error(405)
    do_PUT=do_POST;do_DELETE=do_POST;do_PATCH=do_POST
    def log_message(self,*args):pass

if __name__=='__main__':
    with ThreadingHTTPServer(('127.0.0.1',18769),Handler) as server:server.serve_forever(poll_interval=.5)
