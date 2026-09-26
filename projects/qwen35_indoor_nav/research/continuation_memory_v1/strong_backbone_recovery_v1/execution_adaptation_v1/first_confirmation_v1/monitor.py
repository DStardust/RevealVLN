"""Show the 289-route follow-up separately from the completed discovery80."""
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scope_monitor',HERE.parent/'scope_validation_v1/monitor.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
RUN=HERE/'runs/confirm_001'
CARD='''<section style="background:#234435;margin-bottom:20px"><h1>当前：FIRST固定权重复验 · 剩余289条unseen</h1>
<p>原生 + CURRENT-FIRST/DELTA-FIRST三个种子，共2023次执行，0新训练。发现集80条单独保留；289条此前也有其他结果暴露。</p>
<pre id="followup-live">读取中…</pre><a href="/api/first_confirmation">原始进度</a></section>
<script>async function followupUpdate(){try{let d=await(await fetch('/api/first_confirmation',{cache:'no-store'})).json(),s=d.status||{},r=d.result||{};
let x=[(s.status||'PREPARING')+' · '+(s.phase||''),'完整组 '+(s.sealed_groups||0)+' / 289','GPU会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
if(s.error)x.push('错误：'+s.error);for(let w of s.workers||[]){let p=w.progress||{};x.push('GPU '+w.gpu+' · '+(p.current_id??'加载中')+' '+(p.current_arm||''));}
for(let [a,v] of Object.entries(r.arms||{}))x.push(a+'：'+v.successes+'/'+v.complete+' 已完成；完整SR '+(v.sr===null?'待完成':(100*v.sr).toFixed(2)+'%'));
document.getElementById('followup-live').textContent=x.join('\\n');}catch(e){document.getElementById('followup-live').textContent=String(e);}}followupUpdate();setInterval(followupUpdate,5000);</script>'''


class Handler(previous.Handler):
    def do_GET(self):
        if self.path=='/api/first_confirmation':
            def read(path):return json.loads(path.read_text()) if path.exists() else None
            final=RUN/'unseen/RESULT.json'
            body=json.dumps(dict(status=read(RUN/'STATUS.json'),result=read(final if final.exists() else RUN/'unseen/LIVE_RESULT.json')),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/api/scope_validation' and (previous.RUN/'unseen/RESULT.json').exists():
            body=json.dumps(dict(status=json.loads((previous.RUN/'STATUS.json').read_text()),
                result=json.loads((previous.RUN/'unseen/RESULT.json').read_text())),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/':
            old=previous.previous
            discovery=previous.CARD.replace('下一项：','已完成：').replace('旧评测结束后自动开始。','80/80已完成，结果单独保留。')
            page=old.old.old.PAGE.replace('<section>',CARD+discovery+old.CARD+old.old.CARD+'<section>',1)
            body=page.encode();mime='text/html; charset=utf-8'
        else:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18770),Handler).serve_forever()
