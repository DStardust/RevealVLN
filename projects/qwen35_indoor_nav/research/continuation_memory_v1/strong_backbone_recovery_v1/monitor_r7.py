"""Read-only preservation repair progress, retaining all previous dashboards."""
import json
import importlib.util
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
spec=importlib.util.spec_from_file_location('preservation_previous_monitor',Path(__file__).with_name('monitor_r6.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
u=old.u


CARD='''<section style="margin:20px;padding:20px;background:#193449;color:white;border:2px solid #ffdc83;border-radius:12px">
<h2>当前 · 普通导航保持修复 V3</h2><pre id="preserve-live" style="white-space:pre-wrap">加载…</pre>
<p>训练集普通导航保持 + 合法续接运动监督。先检查记忆仍能完成 FIT 任务，再筛选 DEV；之后固定 unseen 200 只评一次。FIT 拟合不等于任务泛化。</p></section>
<script>async function preserveUpdate(){try{let d=await(await fetch('/api/preservation',{cache:'no-store'})).json(),s=d.status;
let stages={ORDINARY_NATIVE_CAPTURE:'采集普通导航因果特征',BUILD_SHARED_POOLS:'构建共享训练池',VERIFY_PRESERVATION_GRADIENT:'核验保持损失与梯度',SHARED_INITIALIZATION:'共享初始化',MATCHED_PRESERVATION_TRAINING:'匹配训练',FIT_AND_CACHED_DEV_REVIEW:'训练与缓存诊断',DEV_PAIRED_EVALUATION:'DEV 20 条闭环对照',UNSEEN_200_ONCE:'unseen 200 条最终对照',UNSEEN_COMPARISON_COMPLETE:'最终对照完成',REPAIR_SEARCH_COMPLETE:'注册修复尝试已完成'};
let v=[s.status+' · '+(stages[s.phase]||s.phase),'普通轨迹采集 '+(s.captured_recorded||0)+'/100；模型状态已封存 '+(s.captured_sealed||0),'GPU 会话小时 '+Number(s.gpu_hours||0).toFixed(3)];
if(s.trial!=null)v.push('修复强度 '+(s.trial+1)+'/3 · 保持权重 '+s.preservation_weight);
if(d.gradient)v.push('真实保持损失反向：'+d.gradient.queries+' 个动作，梯度范数 '+d.gradient.gradient_norm.toFixed(4)+'（旧失败头诊断，不计训练成果）');
if(d.admission)v.push('共享监督：普通FIT '+d.admission.ordinary.FIT.queries+' 个动作；合法后缀 '+d.admission.dense_queries+' 个动作');
for(let t of d.training)v.push('尝试 '+(Number(t.trial)+1)+' '+t.arm+'：'+t.step+'/1200');
for(let w of s.workers||[])v.push('GPU '+w.gpu+' '+w.name+(w.progress&&w.progress.complete?' · '+w.progress.complete.length+'/'+w.progress.planned.length:''));
for(let e of d.evaluations){v.push(e.label+'：已封存 '+e.complete+'/'+e.planned+' 组');for(let [a,r] of Object.entries(e.arms))v.push(a+' '+r.successes+'/'+e.complete+'，SR '+(e.complete?(100*r.successes/e.complete).toFixed(1)+'%':'待测')+'（当前完整组分母）');}
if(d.result)v.push('判定：'+d.result.status+'；'+d.result.reason);if(s.error)v.push('错误：'+s.error);
document.getElementById('preserve-live').textContent=v.join('\\n');}catch(e){document.getElementById('preserve-live').textContent='读取失败 '+e}}
preserveUpdate();setInterval(preserveUpdate,5000);</script>'''


def payload():
    root=Path(u.read(u.HERE/'PRESERVATION_RUN.json')['path']);read=lambda name:u.read(root/name) if (root/name).exists() else None
    training=[];evaluations=[]
    for path in root.glob('trials/*/training/*/PROGRESS.json'):
        row=u.read(path);training.append(dict(trial=path.parents[2].name,arm=path.parent.name,step=row['step']))
    for path in [*root.glob('trials/*/dev'),root/'unseen']:
        if not (path/'DATA_MANIFEST.json').exists():continue
        groups={}
        for session in (path/'evaluation').glob('*'):
            if not (session/'STATE_SEAL.json').exists():continue
            for complete in session.glob('episodes/*/COMPLETE.json'):
                row=u.read(complete)
                if row['id'] in groups:raise ValueError('DUPLICATE_COMPLETE_GROUP')
                groups[row['id']]=row
        planned=len(u.read(path/'DATA_MANIFEST.json')['episodes'])
        evaluations.append(dict(label='unseen' if path.name=='unseen' else 'DEV 尝试 '+str(int(path.parent.name)+1),complete=len(groups),planned=planned,
            arms={a:dict(successes=sum(bool(g['outcomes'][a]['success']) for g in groups.values())) for a in ('NATIVE','BC','B2','OURS')}))
    return dict(status=read('STATUS.json'),gradient=read('PRESERVATION_GRADIENT.json'),admission=read('data/ADMISSION.json'),
        training=training,evaluations=evaluations,result=read('RESULT.json'))


class Handler(old.Handler):
    def do_GET(self):
        if self.path not in ('/','/api/preservation'):return super().do_GET()
        if self.path=='/api/preservation':body=json.dumps(payload(),ensure_ascii=False).encode();mime='application/json; charset=utf-8'
        else:
            page=(old.previous.base.OLD/'index.html').read_text()
            body=page.replace('<h1>',CARD+old.REPAIR_CARD+old.TRANSFER_CARD+old.CARD+old.previous.base.CARD+'<h1>',1).encode();mime='text/html; charset=utf-8'
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)));self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1]) if len(sys.argv)>1 else 18770),Handler).serve_forever()
