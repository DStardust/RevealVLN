"""Show actual TRAIN recovery collection, gate updates and live unseen results."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer

spec=importlib.util.spec_from_file_location('recovery_previous_monitor',Path(__file__).with_name('monitor_r8.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);u=old.u

CARD='''<section style="margin:20px;padding:20px;background:#163b46;color:white;border:2px solid #65d4bd;border-radius:12px">
<h2>当前 · 补齐真实收益/伤害 → 训练 → unseen</h2>
<pre id="recovery-live" style="white-space:pre-wrap">加载…</pre>
<p>新 TRAIN 400 条路线 × 正常/实际转向前缀 × 四臂，共 3200 次导航。训练只用 TRAIN；完成后自动对固定已暴露 unseen 200 条运行原生、旧方法及可训练的门控方法。不是完整 1839 条；新架构原型不在本轮。</p></section>
<script>async function recoveryUpdate(){try{let d=await(await fetch('/api/intervention_v2',{cache:'no-store'})).json(),s=d.status;
let phase={COLLECT_TRAIN:'采集 TRAIN 正常/偏离恢复对照',FIT_GATES:'训练介入选择器',LIVE_UNSEEN:'真实 unseen 导航',UNSEEN_REVIEW_COMPLETE:'unseen 评测完成',NO_TRAINABLE_GAIN_HARM_GATE:'真实类别仍不足，未训练/未测新策略'};
let v=[s.status+' · '+(phase[s.phase]||s.phase),'当前阶段记录 '+s.recorded_groups+'/'+s.planned_groups+' 组；已封存 '+s.sealed_groups,'累计 GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
for(let w of s.workers||[])v.push((w.gpu==null?'CPU':'GPU '+w.gpu)+' · '+(w.progress?w.progress.complete.length+'/'+w.progress.planned.length:'加载或计算中'));
for(let t of d.training)v.push(t.arm+' 实际训练 '+t.step+'/'+t.planned+' 步');
if(d.counts)for(let [a,r] of Object.entries(d.counts.by_arm_partition))v.push(a+' FIT样本 '+JSON.stringify(r.FIT)+' DEV样本 '+JSON.stringify(r.DEV));
if(d.gates)for(let [a,r] of Object.entries(d.gates.results))v.push(a+' '+r.status+'；实际更新 '+r.actual_steps);
if(d.live){v.push('以下分数属于 '+d.live.split);for(let [a,r] of Object.entries(d.live.arms))v.push(a+' 成功 '+r.successes+'/'+r.planned+'（已完成 '+r.complete+'）'+(r.sr==null?'，未完成分母':'；SR '+(100*r.sr).toFixed(1)+'%'));}
if(s.error)v.push('错误：'+s.error);document.getElementById('recovery-live').textContent=v.join('\\n');}catch(e){document.getElementById('recovery-live').textContent='读取失败 '+e}}
recoveryUpdate();setInterval(recoveryUpdate,5000);</script>'''


class Handler(old.Handler):
    def do_GET(self):
        if self.path not in ('/','/api/intervention_v2'):return super().do_GET()
        if self.path=='/api/intervention_v2':
            root=Path(u.read(u.HERE/'INTERVENTION_V2_RUN.json')['path']);status=u.read(root/'STATUS.json')
            def read_optional(path):return u.read(path) if path.exists() else None
            live=read_optional(root/'unseen/LIVE_RESULT.json') or read_optional(root/'train/LIVE_RESULT.json')
            payload=dict(status=status,training=[u.read(p) for p in root.glob('gates/*/PROGRESS.json')],
                counts=read_optional(root/'gate_data/COUNTS.json'),gates=read_optional(root/'GATE_REVIEW.json'),
                live=live,result=read_optional(root/'RESULT.json'))
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        else:
            r7=old.old;r6=r7.old;page=(r6.previous.base.OLD/'index.html').read_text()
            cards=CARD+old.CARD+r7.CARD+r6.REPAIR_CARD+r6.TRANSFER_CARD+r6.CARD+r6.previous.base.CARD
            body=page.replace('<h1>',cards+'<h1>',1).encode();mime='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
