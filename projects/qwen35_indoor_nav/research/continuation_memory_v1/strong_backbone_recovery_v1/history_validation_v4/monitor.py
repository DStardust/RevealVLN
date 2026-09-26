"""Extend the existing monitor, keeping its historical pages and APIs."""
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirmation_monitor',HERE.parent/'recovery_confirmation_v2/monitor.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
RUN = HERE/'runs/validation_001'

PAGE = '''<!doctype html><meta charset="utf-8"><title>Q35N 历史机制与新增路线复验</title>
<style>body{font:16px system-ui;background:#101827;color:#edf2f7;margin:24px}section{background:#15372d;border:1px solid #68d391;padding:22px;border-radius:12px}pre{white-space:pre-wrap;line-height:1.6}a{color:#90cdf4}iframe{width:100%;height:1800px;border:0}</style>
<section><h1>当前：冻结六模型，新增路线复验</h1><pre id="live">加载中…</pre>
<p>旧200条结果单列。新增路线按完整物理路线排除旧清单，不选择容易样本。CPU 记忆干预是离线诊断，不计作导航成功率。本轮不训练。</p>
<a href="/previous">此前完整训练与评测监控</a></section>
<script>async function update(){try{let d=await(await fetch('/api/history_validation',{cache:'no-store'})).json(),s=d.status;
let lines=[s.status+' · '+s.phase,'记录 '+(s.recorded_groups||0)+' / '+s.planned_groups+' 组，已封存 '+(s.sealed_groups||0),'计划真实执行 '+s.planned_executions+' 次；GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
if(d.mechanism)lines.push('CPU 机制检验：'+d.mechanism.status+' '+d.mechanism.model_rows+'/'+d.mechanism.planned_model_rows+' 模型×轨迹');
for(let w of s.workers||[]){let p=w.progress;lines.push((w.gpu==null?'CPU':'GPU '+w.gpu)+'：'+(p?p.complete.length+'/'+p.planned.length+' 组':'加载/检验中'));}
if(d.result){for(let [a,v] of Object.entries(d.result.arms)){lines.push(a+' '+v.successes+'/'+v.planned+'，完成 '+v.complete+(v.sr==null?'，全分母SR待完成':'，SR '+(100*v.sr).toFixed(2)+'%'));}}
if(s.error)lines.push('错误：'+s.error);document.getElementById('live').textContent=lines.join('\\n');}catch(e){document.getElementById('live').textContent='读取失败 '+e}}
update();setInterval(update,5000);</script><iframe src="/previous" title="历史进度"></iframe>'''


class Handler(old.Handler):
    def do_GET(self):
        if self.path=='/previous':
            self.path='/'
            return super().do_GET()
        if self.path=='/api/history_validation':
            def read(name):
                p=RUN/name
                return old.u.read(p) if p.exists() else None
            body=json.dumps(dict(status=read('STATUS.json'),mechanism=read('mechanism/STATUS.json'),
                result=read('RESULT.json') or read('LIVE_RESULT.json')),ensure_ascii=False).encode()
            mime='application/json; charset=utf-8'
        elif self.path=='/':
            body=PAGE.encode()
            mime='text/html; charset=utf-8'
        else:
            return super().do_GET()
        self.send_response(200)
        self.send_header('Content-Type',mime)
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):
            pass


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
