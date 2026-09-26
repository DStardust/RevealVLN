"""Show active model optimization above the unchanged old evaluation cards."""
import importlib.util
import json
from pathlib import Path
from http.server import ThreadingHTTPServer

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('first_monitor',HERE.parent/'first_confirmation_v1/monitor.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
RUN=HERE/'runs/alignment_001'
CARD='''<section style="background:#234435;margin-bottom:20px"><h1>当前改进：FIRST 训练与执行位置对齐</h1>
<p>新训练 CURRENT / DELTA × 三个种子，各3000步。完整物理历史保留，仅训练实际干预位置。随后自动进行新旧模型 unseen80 对照，共1040次执行。</p>
<pre id="aligned-live">读取中…</pre><a href="/api/first_alignment">训练与评测原始进度</a></section>
<script>async function alignedUpdate(){try{let d=await(await fetch('/api/first_alignment',{cache:'no-store'})).json(),s=d.status||{},r=d.result||{};
let x=[(s.status||'PREPARING')+' · '+(s.phase||''),'GPU会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let [a,v] of Object.entries(s.models||{})){v=v||{};x.push(a+'：'+(v.step??v.completed_steps??0)+' / 3000 '+(v.status||'待启动'));}
x.push('已封存完整评测组 '+(s.sealed_groups||0)+' / 80');if(s.error)x.push('错误：'+s.error);
for(let [a,v] of Object.entries(r.arms||{}))x.push(a+'：成功 '+v.successes+'/'+v.complete+'；完整SR '+(v.sr==null?'待完成':(100*v.sr).toFixed(2)+'%'));
document.getElementById('aligned-live').textContent=x.join('\\n');}catch(e){document.getElementById('aligned-live').textContent=String(e);}}alignedUpdate();setInterval(alignedUpdate,5000);</script>'''


class Handler(previous.Handler):
    def do_GET(self):
        if self.path=='/api/first_alignment':
            def read(p):return json.loads(p.read_text()) if p.exists() else None
            final=RUN/'unseen/RESULT.json'
            body=json.dumps(dict(status=read(RUN/'STATUS.json'),result=read(final if final.exists() else RUN/'unseen/LIVE_RESULT.json')),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/':
            # Reuse the existing page and all historical endpoints.
            scope=previous.previous;adapt=scope.previous
            discovery=scope.CARD.replace('下一项：','已完成：').replace('旧评测结束后自动开始。','80/80已完成，结果单独保留。')
            page=adapt.old.old.PAGE.replace('<section>',CARD+previous.CARD+discovery+adapt.CARD+adapt.old.CARD+'<section>',1)
            body=page.encode();mime='text/html; charset=utf-8'
        else:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',18770),Handler).serve_forever()
