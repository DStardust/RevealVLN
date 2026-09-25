"""Read-only progress for the recovery-action architecture comparison."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_monitor',HERE.parent/'recovery_action_v1/monitor.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);u=old.u

CARD='''<section style="margin:20px;padding:20px;background:#163629;color:white;border:2px solid #6ee7b7;border-radius:12px">
<h2>当前 · 三种子复验：新增历史是否有用</h2><pre id="confirmation-live" style="white-space:pre-wrap">加载…</pre>
<p>42 / 43 / 44 三个种子；CONCAT 保留新增递归历史，LOCAL 仅编码当前观测。基座自带历史均保留。共享原数据、初始化、样本顺序和 3000 步；完整 unseen200，每条件原生＋六个模型。已暴露开发结果，不是盲测。</p></section>
<script>async function confirmationUpdate(){try{let d=await(await fetch('/api/confirmation',{cache:'no-store'})).json(),s=d.status;
let v=[s.status+' · '+s.phase,'当前阶段 '+(s.recorded_groups||0)+'/'+(s.planned_groups||0)+' 组；已封存 '+(s.sealed_groups||0),'GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let t of d.training)v.push(t.arm+'：'+t.status+' '+t.step+'/'+t.planned+'步');
for(let w of s.workers||[]){let p=w.progress;v.push('GPU '+w.gpu+' · '+(p?(p.step!=null?p.step+'/'+p.planned:p.complete.length+'/'+p.planned.length):'模型加载/初始化'));}
for(let [name,r] of Object.entries(d.results)){if(!r)continue;v.push(name+'：'+r.complete_groups+'/'+r.planned_groups+' 完整组');for(let [a,z] of Object.entries(r.arms))v.push(a+'：'+z.successes+'/'+z.planned+(z.sr==null?'（尚未完成）':' SR '+(100*z.sr).toFixed(1)+'%'));}
if(s.error)v.push('错误：'+s.error);document.getElementById('confirmation-live').textContent=v.join('\\n');}catch(e){document.getElementById('confirmation-live').textContent='读取失败 '+e}}
confirmationUpdate();setInterval(confirmationUpdate,5000);</script>'''



class Handler(old.Handler):
    def do_GET(self):
        def read(p):return u.read(p) if p.exists() else None
        if self.path=='/api/confirmation':
            root=HERE/'runs/confirmation_001';training=[]
            for arm in u.read(root/'PROTOCOL.json')['models']:
                folder=root/'training'/arm;p=read(folder/'PROGRESS.json') or {}
                training.append(dict(arm=arm,status='COMPLETE' if (folder/'RESULT.json').exists() else p.get('status','PENDING'),step=p.get('step',0),planned=3000))
            payload=dict(status=u.read(root/'STATUS.json'),training=training,
                results={name:read(root/name/'RESULT.json') or read(root/name/'LIVE_RESULT.json') for name in ('dev','unseen')})
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/':
            r9=old.old.old;r8=r9.old;r7=r8.old;r6=r7.old
            page=(r6.previous.base.OLD/'index.html').read_text()
            cards=CARD+old.CARD.replace('当前 ·','上一轮 ·')+r9.CARD.replace('当前 ·','上一轮 ·')+r8.CARD+r7.CARD+r6.REPAIR_CARD+r6.TRANSFER_CARD+r6.CARD+r6.previous.base.CARD
            body=page.replace('<h1>',cards+'<h1>',1).encode();mime='text/html; charset=utf-8'
        else:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
