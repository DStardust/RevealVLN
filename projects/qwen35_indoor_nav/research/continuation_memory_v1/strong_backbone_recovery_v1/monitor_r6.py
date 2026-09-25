"""Read-only strong-backbone progress on the user's existing dashboard."""
import importlib.util
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
_spec=importlib.util.spec_from_file_location('strong_monitor_common',Path(__file__).with_name('common.py'))
u=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(u)
spec=importlib.util.spec_from_file_location('strong_previous_monitor',u.PROJECT/'sft_acceptance/ordinary_branch_decision_v21/monitor_r1.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

CARD='''<section style="margin:20px;padding:20px;background:#173e36;color:#eef7ff;border:2px solid #68dfb0;border-radius:12px">
<h2>公开强导航模型 · StreamVLN</h2><p id="strong-status">读取状态…</p><pre id="strong-detail" style="white-space:pre-wrap"></pre>
<p>先核验零增量不改变原模型，再完成原生100条开发导航。原模型成绩与新增记忆方法收益分开显示。</p></section>
<script>async function strongUpdate(){try{let d=await(await fetch('/api/strong',{cache:'no-store'})).json(),s=d.status;
document.getElementById('strong-status').textContent=d.run+' · '+s.status+' · '+s.phase;
let lines=['已记录原生导航：'+(s.recorded_native||s.completed_native||0)+'/100；完成会话封存：'+(s.completed_native||0),
'累计GPU会话小时：'+Number(s.gpu_hours||0).toFixed(3), '新增方法训练：'+(d.method&&d.method.training.filter(t=>t.status==='TRAINING_COMPLETE').length===3?'三臂均已完成1200步':d.method&&d.method.training.length?'进行中':'尚未启动'),
'新增方法收益：'+(s.method_benefit||'未测量')];
for(let w of s.workers||[])lines.push('GPU'+w.gpu+' '+w.phase+' · '+(w.progress? w.progress.complete.length+'/'+w.progress.planned.length:'正在加载或启动'));
if(d.method){let m=d.method;lines.push('记忆对照：'+(m.recovery?m.recovery.status:m.status.status)+' · '+(m.recovery?'复核已修复，CHECK缺样本':m.status.phase||m.status.error||''));lines.push('记忆对照GPU会话小时：'+Number(m.status.gpu_hours||0).toFixed(3));for(let f of m.features)lines.push(f.family+' 特征 '+f.complete+'/'+f.planned);for(let t of m.training)lines.push(t.arm+' 训练 '+t.step+'/'+t.planned);if(m.recovery){for(let r of m.recovery_result.results.FIT.arms)lines.push('FIT拟合动作 '+r.arm+' '+r.correct+'/'+r.total);lines.push('CHECK缺少可用样本；闭环增量未测量');}else if(m.review)lines.push('最终动作诊断已完成；闭环增量尚未测量');}if(d.unseen){let n=d.unseen.status;lines.push('━━ 官方unseen固定子集 · 原生StreamVLN ━━');lines.push(n.status+' · '+(n.recorded_native||n.completed_native||0)+'/'+n.planned_native+'条；10屋/200条不同路线');for(let w of n.workers||[])lines.push('GPU'+w.gpu+' '+(w.progress?w.progress.complete.length+'/'+w.progress.planned.length:'模型加载中'));if(d.unseen.result)lines.push('unseen子集SR：'+(100*d.unseen.result.sr).toFixed(1)+'%（不是完整1839条）');if(n.error)lines.push('错误：'+n.error);}if(d.data_status){let c=d.data_status;lines.push('移动历史重验：'+c.status+' '+(c.completed_traces||0)+'/'+(c.planned_traces||72));if(c.error)lines.push('数据错误：'+c.error);}if(s.error)lines.push('错误：'+s.error);if(d.result)lines.push('本机原生SR：'+(100*d.result.sr).toFixed(1)+'%（'+d.result.complete+'/'+d.result.planned+'；不计作我们的增量）');
document.getElementById('strong-detail').textContent=lines.join('\\n');}catch(e){document.getElementById('strong-status').textContent='状态读取失败：'+e;}}
strongUpdate();setInterval(strongUpdate,5000);</script>'''

TRANSFER_CARD='<section style="margin:20px;padding:20px;background:#18354b;color:#fff;border:2px solid #75bfff;border-radius:12px"><h2>固定 unseen 200 条 · 加入记忆后的普通导航对照</h2><pre id="transfer-live" style="white-space:pre-wrap">读取…</pre><p>原生 / BC / B2 / Ours，同一模型进程逐路线比较。普通能力保持测试，不代替历史恢复任务。</p></section><script>async function transferUpdate(){try{let d=await(await fetch(\'/api/transfer\',{cache:\'no-store\'})).json(),s=d.status;let v=[s.status+\' · \'+s.phase,\'已完成四臂组：\'+(s.recorded_groups||0)+\'/200；已封存：\'+(s.sealed_groups||0),\'GPU会话小时：\'+Number(s.gpu_hours||0).toFixed(3)];if(d.diagnosis)v.unshift(\'事后纠错：动作接口位置错误；记忆增量未作用于真正动作，当前同分不能作为方法有效或能力保持证据。\');for(let w of s.workers||[])v.push(\'GPU\'+w.gpu+\' · \'+(w.progress?w.progress.complete.length+\'/\'+w.progress.planned.length:\'加载/核验中\'));if(d.result)for(let [a,r] of Object.entries(d.result.arms))v.push(a+\'：\'+r.successes+\'/\'+r.complete+\'，SR \'+(r.sr_completed==null?\'待测\':(100*r.sr_completed).toFixed(1)+\'%\')+\'（封存分母；完整计划每臂200）\');if(s.error)v.push(\'错误：\'+s.error);document.getElementById(\'transfer-live\').textContent=v.join(\'\\n\')}catch(e){document.getElementById(\'transfer-live\').textContent=\'读取失败 \'+e}}transferUpdate();setInterval(transferUpdate,5000)</script>'

REPAIR_CARD='<section style="margin:20px;padding:20px;background:#20402f;color:white;border:2px solid #73de9c;border-radius:12px"><h2>当前运行 · 真实动作位置修复 V2</h2><pre id="repair-live" style="white-space:pre-wrap">读取状态…</pre><p>顺序：非零动作接口核验 → 重建正确特征 → 三臂匹配训练 → 固定200条对照。接口修复、训练拟合与导航收益分别显示。</p></section><script>async function repairUpdate(){try{let d=await(await fetch(\'/api/repair\',{cache:\'no-store\'})).json(),s=d.status;let stages={LIVE_ACTION_PROBE:\'真实动作接口检查\',CORRECTED_FEATURES:\'重建动作位置特征\',SHARED_INITIALIZATION:\'共享初始化\',MATCHED_TRAINING:\'三臂匹配训练\',TRAINING_REVIEW:\'训练诊断\',EVALUATION_FIRST_FIVE:\'首5条四臂对照\',EVALUATION_REMAINING:\'剩余195条四臂对照\',FINAL_REVIEW:\'最终复核\',MATCHED_COMPARISON_COMPLETE:\'对照完成\'};let v=[s.status+\' · \'+(stages[s.phase]||s.phase),\'GPU会话小时：\'+Number(s.gpu_hours||0).toFixed(3)];if(d.probe)v.push(\'已确认：非零增量改变了环境实际执行动作（诊断，不计方法收益）\');for(let f of d.features)v.push(f.family+\' 特征 \'+f.complete+\'/\'+f.planned);for(let t of d.training)v.push(t.arm+\' 训练 \'+t.step+\'/\'+t.planned+(t.finished?\' 已完成\':\'\'));for(let w of s.workers||[])v.push(\'GPU \'+w.gpu+\' · \'+w.name);if(d.training_review)for(let a of d.training_review.results.FIT.arms)v.push(\'修复后 FIT 动作拟合 \'+a.arm+\'：\'+a.correct+\'/\'+a.total);if(d.ordinary){v.push(\'新普通导航：\'+(s.recorded_groups||0)+\'/200 四臂组；已封存 \'+d.ordinary.complete_groups);for(let [a,r] of Object.entries(d.ordinary.arms))v.push(a+\' \'+r.successes+\'/\'+r.complete+\'，SR \'+(r.sr_completed==null?\'待测\':(100*r.sr_completed).toFixed(1)+\'%\'));}if(s.error)v.push(\'错误：\'+s.error);document.getElementById(\'repair-live\').textContent=v.join(\'\\n\')}catch(e){document.getElementById(\'repair-live\').textContent=\'读取失败 \'+e}}repairUpdate();setInterval(repairUpdate,5000)</script>'

class Handler(previous.Handler):
    def do_GET(self):
        if self.path not in ('/','/api/strong','/api/transfer','/api/repair'):return super().do_GET()
        if self.path=='/api/repair':
            root=Path(u.read(u.HERE/'REPAIR_RUN.json')['path'])
            payload=dict(status=u.read(root/'STATUS.json'),features=[u.read(p) for p in (root/'features').glob('*/STATUS.json')],training=[dict(u.read(p),finished=(p.parent/'FINAL.pt').exists()) for p in (root/'training').glob('*/PROGRESS.json')],probe=bool(list(root.glob('probe/sessions/*/episodes/0/NONZERO_PROBE/PROBE_EXECUTION.json'))),training_review=u.read(root/'TRAINING_REVIEW.json') if (root/'TRAINING_REVIEW.json').exists() else None,ordinary=u.read(root/'ordinary/LIVE_RESULT.json') if (root/'ordinary/LIVE_RESULT.json').exists() else None)
            body=json.dumps(payload,ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/api/transfer':
            folder=Path(u.read(u.HERE/'TRANSFER_RUN.json')['path'])
            correction=u.HERE/'TRANSFER_POSTRUN_DIAGNOSIS.json'
            diagnosis=u.read(u.read(correction)['path']) if correction.exists() and u.read(correction)['run']==str(folder) else None
            body=json.dumps(dict(diagnosis=diagnosis,status=u.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else {'status':'STARTING'},result=u.read(folder/'LIVE_RESULT.json') if (folder/'LIVE_RESULT.json').exists() else None),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        elif self.path=='/api/strong':
            run=u.HERE/'runs'/(u.HERE/'LAST_RUN.txt').read_text().strip()
            method=None
            if (u.HERE/'METHOD_RUN.json').exists():
                folder=Path(u.read(u.HERE/'METHOD_RUN.json')['path'])
                method=dict(status=u.read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else {'status':'STARTING'},
                    features=[u.read(p) for p in sorted((folder/'features').glob('*/STATUS.json'))],
                    training=[dict(u.read(p),**({'status':'TRAINING_COMPLETE'} if (p.parent/'RESULT.json').exists() and u.read(p.parent/'RESULT.json')['status']=='TRAINING_COMPLETE' else {})) for p in sorted((folder/'training').glob('*/PROGRESS.json'))],review=(folder/'ACTION_REVIEW.json').exists())
            if method is not None and (u.HERE/'REVIEW_RECOVERY.json').exists():
                recovered=u.read(u.HERE/'REVIEW_RECOVERY.json')
                if recovered['run']==folder.name:
                    method['recovery']=u.read(recovered['status_path']);method['recovery_result']=u.read(recovered['review_path'])
            unseen=None
            if (u.HERE/'UNSEEN_RUN.json').exists():
                unseen_folder=Path(u.read(u.HERE/'UNSEEN_RUN.json')['path'])
                unseen=dict(status=u.read(unseen_folder/'STATUS.json'),result=u.read(unseen_folder/'RESULT.json') if (unseen_folder/'RESULT.json').exists() else None)
            body=json.dumps(dict(unseen=unseen,method=method,run=run.name,status=u.read(run/'STATUS.json'),data_status=u.read(u.HERE/'data_runs/moving_003/STATUS.json') if (u.HERE/'data_runs/moving_003/STATUS.json').exists() else None,result=u.read(run/'RESULT.json') if (run/'RESULT.json').exists() else None),ensure_ascii=False).encode()
            mime='application/json; charset=utf-8'
        else:
            page=(previous.base.OLD/'index.html').read_text()
            body=page.replace('<h1>',REPAIR_CARD+TRANSFER_CARD+CARD+previous.base.CARD+'<h1>',1).encode();mime='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
