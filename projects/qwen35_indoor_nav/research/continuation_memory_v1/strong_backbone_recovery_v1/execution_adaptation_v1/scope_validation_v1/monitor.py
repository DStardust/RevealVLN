"""Keep current evaluation visible and add the registered next scope test."""
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('adaptation_monitor',HERE.parent/'operation_v1/monitor.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.RUN=HERE.parent/'runs/experiment_002'
RUN=HERE/'runs/scope_001'
CARD='''<section style="background:#243948;margin-bottom:20px"><h1>下一项：首动作纠偏 vs 整段纠偏</h1>
<p>同一冻结权重，CURRENT/DELTA 各三种子，80条固定unseen × 13个版本。0次新训练；旧评测结束后自动开始。</p>
<pre id="scope-live">读取中…</pre><a href="/api/scope_validation">原始进度</a></section>
<script>async function scopeUpdate(){try{let d=await(await fetch('/api/scope_validation',{cache:'no-store'})).json(),s=d.status||{},r=d.result||{};
let x=[(s.status||'PREPARING')+' · '+(s.phase||''),'完整组 '+(s.sealed_groups||0)+' / 80','GPU会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
if(s.predecessor_sealed_groups!==undefined)x.push('正在等待旧评测：'+s.predecessor_sealed_groups+' / 369');
if(s.error)x.push('错误：'+s.error);for(let w of s.workers||[]){let p=w.progress||{};x.push('GPU '+w.gpu+' · '+(p.current_id??'加载中')+' '+(p.current_arm||''));}
for(let [a,v] of Object.entries(r.arms||{}))x.push(a+'：'+v.successes+'/'+v.complete+' 已完成；完整SR '+(v.sr===null?'待完成':(100*v.sr).toFixed(2)+'%'));
document.getElementById('scope-live').textContent=x.join('\\n');}catch(e){document.getElementById('scope-live').textContent=String(e);}}scopeUpdate();setInterval(scopeUpdate,5000);</script>'''


class Handler(previous.Handler):
    def do_GET(self):
        if self.path=='/api/scope_validation':
            def read(path):return json.loads(path.read_text()) if path.exists() else None
            body=json.dumps(dict(status=read(RUN/'STATUS.json'),result=read(RUN/'unseen/LIVE_RESULT.json')),ensure_ascii=False).encode()
            mime='application/json; charset=utf-8'
        elif self.path=='/':
            body=previous.old.old.PAGE.replace('<section>',CARD+previous.CARD+previous.old.CARD+'<section>',1).encode()
            mime='text/html; charset=utf-8'
        else:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18770),Handler).serve_forever()
