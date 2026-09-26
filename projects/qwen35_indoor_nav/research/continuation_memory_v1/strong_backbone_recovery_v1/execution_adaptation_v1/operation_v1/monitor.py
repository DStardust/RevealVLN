"""Add current training/unseen progress, retaining every previous monitor route."""
import argparse
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
spec = importlib.util.spec_from_file_location('recapture_monitor', BASE / 'execution_data_v1/recapture_v1/monitor.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
RUN = HERE.parent / 'runs/experiment_001'
CARD = r'''<section style="background:#193d36;margin-bottom:20px">
<h1>当前：补采完成 → 同容量新架构训练 → unseen评测</h1>
<pre id="adapt-live">加载中…</pre>
<p>CURRENT：当前特征写入；DELTA：观测变化写入。三种子各3000步，冻结底模。旧正向候选保留。</p>
<a href="/api/execution_adaptation">当前原始进度</a></section>
<script>
async function adaptationUpdate(){try{
let d=await(await fetch('/api/execution_adaptation',{cache:'no-store'})).json(),s=d.status||{};
let a=[(s.status||'NOT_STARTED')+' · '+(s.phase||'PREPARING'),
'最终模型 '+(s.complete_models||[]).length+' / 6',
'unseen完整组 '+(s.sealed_groups||0)+' / 369；记录组 '+(s.recorded_groups||0),
'GPU会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let [name,v] of Object.entries(s.models||{})){a.push(name+'：'+(v?(v.completed_steps??v.step??0):0)+' / 3000'+(v?' · '+v.status:''));}
for(let w of s.workers||[]){let q=w.progress||{};a.push('GPU '+w.gpu+' · PID '+w.pid+' · '+(q.current_arm||'')+' '+(q.current_id??q.step??'加载/运行中'));}
if(s.error)a.push('停止原因：'+s.error);
let r=d.evaluation||{};
for(let [name,v] of Object.entries(r.arms||{})){a.push(name+'：'+v.successes+'/'+v.planned+'；完整SR '+(v.sr===null?'待完成':(100*v.sr).toFixed(2)+'%'));}
document.getElementById('adapt-live').textContent=a.join('\n');
}catch(e){document.getElementById('adapt-live').textContent='读取失败 '+e;}}
adaptationUpdate();setInterval(adaptationUpdate,5000);
</script>'''


def payload():
    def read(path):
        try: return json.loads(path.read_text())
        except FileNotFoundError: return None
    return dict(status=read(RUN/'STATUS.json'),protocol=read(RUN/'PROTOCOL.json'),
        result=read(RUN/'RESULT.json'),evaluation=read(RUN/'unseen/LIVE_RESULT.json'))


class Handler(old.Handler):
    def do_GET(self):
        if self.path == '/api/execution_adaptation':
            body=json.dumps(payload(),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path == '/':
            page=old.old.PAGE.replace('<section>',CARD+old.CARD+'<section>',1)
            body=page.encode();mime='text/html; charset=utf-8'
        else:
            return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('port',type=int,nargs='?',default=18770)
    parser.add_argument('--run',type=Path,default=RUN);a=parser.parse_args();RUN=a.run.resolve()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
