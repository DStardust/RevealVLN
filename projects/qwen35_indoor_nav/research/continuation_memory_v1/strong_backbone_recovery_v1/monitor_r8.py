"""Add TRAIN intervention calibration to the existing read-only dashboard."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
spec=importlib.util.spec_from_file_location('intervention_previous_monitor',Path(__file__).with_name('monitor_r7.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);u=old.u

CARD='''<section style="margin:20px;padding:20px;background:#3a2d4c;color:white;border:2px solid #d0a1ff;border-radius:12px">
<h2>当前 · 学习何时介入</h2><pre id="intervention-live" style="white-space:pre-wrap">加载…</pre>
<p>200 条新 TRAIN 路线 / 800 次真实导航；160 FIT、40 DEV，按房屋隔离。原生与记忆头冻结，采集干预收益和伤害，然后训练一次性后续策略选择器。旧 unseen 不进入训练。</p></section>
<script>async function interventionUpdate(){try{let d=await(await fetch('/api/intervention',{cache:'no-store'})).json(),s=d.status;
let phase={COLLECT_TRAIN_PAIRS:'采集真实干预对照',CPU_INTERVENTION_SELECTOR_TRAINING:'拟合干预选择器',CALIBRATION_COMPLETE:'校准完成'};
let v=[s.status+' · '+(phase[s.phase]||s.phase),'完整四臂组 '+(s.recorded_groups||0)+'/200；参数状态已封存 '+(s.sealed_groups||0),'GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let w of s.workers||[])v.push((w.gpu==null?'CPU':'GPU '+w.gpu)+' · '+(w.progress&&w.progress.complete?w.progress.complete.length+'/'+w.progress.planned.length:'加载或计算中'));
for(let t of d.training)v.push(t.arm+' 选择器训练 '+t.step+'/'+t.planned);
if(d.counts)for(let [a,r] of Object.entries(d.counts.by_arm_partition))v.push(a+' FIT '+JSON.stringify(r.FIT)+'；DEV '+JSON.stringify(r.DEV));
if(d.review)for(let [a,r] of Object.entries(d.review.results)){v.push(a+' '+r.status);if(r.planned)v.push('DEV已执行分支选择：原生 '+r.native_success+'/'+r.planned+'，不筛选 '+r.ungated_success+'/'+r.planned+'，选择后 '+r.selected_success+'/'+r.planned+'；尚非在线门控或unseen结果');}
if(s.error)v.push('错误：'+s.error);document.getElementById('intervention-live').textContent=v.join('\\n');}catch(e){document.getElementById('intervention-live').textContent='读取失败 '+e}}
interventionUpdate();setInterval(interventionUpdate,5000);</script>'''


class Handler(old.Handler):
    def do_GET(self):
        if self.path not in ('/','/api/intervention'):return super().do_GET()
        if self.path=='/api/intervention':
            root=Path(u.read(u.HERE/'INTERVENTION_RUN.json')['path'])
            payload=dict(status=u.read(root/'STATUS.json'),training=[u.read(p) for p in root.glob('gates/*/PROGRESS.json')],
                counts=u.read(root/'gate_data/COUNTS.json') if (root/'gate_data/COUNTS.json').exists() else None,
                review=u.read(root/'GATE_REVIEW.json') if (root/'GATE_REVIEW.json').exists() else None)
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        else:
            r6=old.old;page=(r6.previous.base.OLD/'index.html').read_text()
            body=page.replace('<h1>',CARD+old.CARD+r6.REPAIR_CARD+r6.TRANSFER_CARD+r6.CARD+r6.previous.base.CARD+'<h1>',1).encode();mime='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
