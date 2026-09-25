"""Read-only progress for the recovery-action architecture comparison."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_monitor',HERE.parent/'monitor_r10.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);u=old.u

CARD='''<section style="margin:20px;padding:20px;background:#153b49;color:white;border:2px solid #6ee7b7;border-radius:12px">
<h2>当前 · 真实恢复动作 → 两种纠错结构 → unseen</h2>
<pre id="action-live" style="white-space:pre-wrap">加载…</pre>
<p>TRAIN 失败历史 174 条件、原生成功 160 条件。CONCAT 与 EVIDENCE 共享数据、初始化和 3000 步训练；之后同轮原生＋两种新头测试固定 unseen200。旧权重不续训、教师不进入测试。贡献待实际结果验证。</p></section>
<script>async function actionUpdate(){try{let d=await(await fetch('/api/recovery_action',{cache:'no-store'})).json(),s=d.status;
let phases={COLLECT:'执行真实恢复教师与成功轨迹回放',FEATURES:'真实基座前向与因果特征缓存',PREPARE_TRAIN:'整理共享训练池',TRAIN:'两种纠错结构训练',DEV:'TRAIN留出屋诊断',UNSEEN:'unseen真实配对测试',REVIEW_COMPLETE:'评测完成'};
let v=[s.status+' · '+(phases[s.phase]||s.phase),'当前阶段 '+(s.recorded_groups||0)+'/'+(s.planned_groups||334)+' 组；封存 '+(s.sealed_groups||0),'GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let w of s.workers||[]){let p=w.progress;v.push('GPU '+w.gpu+' · '+(p?(p.step!=null?p.step+'/'+p.planned:p.complete.length+'/'+p.planned.length):'加载/初始化'));}
if(d.collection)v.push('采集准入 '+d.collection.admitted+'/'+d.collection.planned+'；未准入保留记录 '+d.collection.not_admitted.length);
for(let t of d.training)v.push(t.arm+' 更新 '+t.step+'/'+t.planned+'；恢复动作准确率 '+(100*t.recovery_accuracy).toFixed(1)+'%（训练诊断）');
for(let [name,r] of Object.entries(d.results)){if(!r)continue;v.push(name+' '+r.complete_groups+'/'+r.planned_groups+'组');for(let [a,z] of Object.entries(r.arms))v.push(a+' 成功 '+z.successes+'/'+z.planned+(z.sr==null?'，未完成分母':'；SR '+(100*z.sr).toFixed(1)+'%'));}
if(s.error)v.push('错误：'+s.error);document.getElementById('action-live').textContent=v.join('\\n');}catch(e){document.getElementById('action-live').textContent='读取失败 '+e}}
actionUpdate();setInterval(actionUpdate,5000);</script>'''


class Handler(old.Handler):
    def do_GET(self):
        def read(p):return u.read(p) if p.exists() else None
        if self.path=='/api/recovery_action':
            root=Path(u.read(HERE.parent/'RECOVERY_ACTION_RUN.json')['path'])
            payload=dict(status=u.read(root/'STATUS.json'),collection=read(root/'COLLECTION_RESULT.json'),
                training=[u.read(p) for p in root.glob('training/*/PROGRESS.json')],
                results={name:read(root/name/'RESULT.json') or read(root/name/'LIVE_RESULT.json') for name in ('dev','unseen')})
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/api/intervention_v2':
            root=Path(u.read(HERE.parent/'INTERVENTION_V2_RUN.json')['path'])
            payload=dict(status=u.read(root/'STATUS.json'),training=[u.read(p) for p in root.glob('gates/*/PROGRESS.json')],
                counts=read(root/'gate_data/COUNTS.json'),gates=read(root/'GATE_REVIEW.json'),
                live=read(root/'RESULT.json') or read(root/'unseen/LIVE_RESULT.json'),result=read(root/'RESULT.json'))
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/':
            r9=old.old;r8=r9.old;r7=r8.old;r6=r7.old
            page=(r6.previous.base.OLD/'index.html').read_text()
            cards=CARD+r9.CARD.replace('当前 ·','上一轮 ·')+r8.CARD+r7.CARD+r6.REPAIR_CARD+r6.TRANSFER_CARD+r6.CARD+r6.previous.base.CARD
            body=page.replace('<h1>',cards+'<h1>',1).encode();mime='text/html; charset=utf-8'
        else:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
