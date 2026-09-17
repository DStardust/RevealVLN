"""Same 18766 charts, new stage only; read-only source and explicit audit status."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
PARENT=HERE.parent/'monitor_charts_recovery_v1/server.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='0af76a64ae5311a68628c4011f2cfc38da34c7afb7b543e874df8b842cdf8ae0'
s=importlib.util.spec_from_file_location('unchanged_recovery_charts',PARENT)
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
r.RECOVERY=HERE.parent/'ordinary_expanded_v1';r.FORMAL=r.RECOVERY/'formal'
_old_points=r.points_for_segments
r.points_for_segments=lambda segments:_old_points([(name,rows) for name,rows in segments if name!='legacy_v3'])


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def collect(include_gpu=True):
    d=r.collect(include_gpu=include_gpu)
    audit=read(LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/PROGRESS.json')
    d.update(monitor_version='ordinary_expanded_v1',legacy_history_available=False,
             epoch_total_decisions=2650347,data_preparation=audit,
             stage='EXPANDED_ORDINARY_BC_PILOT',source_checkpoint_updates=51301,
             full_benchmark_reference=dict(checkpoint_updates=41800,episodes=1839,sr=405/1839,spl=.18682570665384785),
             eta_note='New stage only: max 4000 updates / 350000 decisions / 1 hour; no automatic long training.')
    status=d['status']['data'] or {}
    if not status:
        d['display_state']='DATA_AUDIT' if not audit.get('training_admitted') else 'PREPARING_TRAINING_INDEX'
    progress=d['progress']['data'] or {}
    cursor=progress.get('cursor',{})
    speed=d.get('steady_decisions_per_second')
    if status.get('status')=='TRAINING' and speed and progress.get('throughput',0)>0:
        recent=r.b.tail_records(Path(status['run_dir'])/'PROGRESS.jsonl')['records'][-11:]
        if len(recent)>1:
            dt=recent[-1]['unix']-recent[0]['unix'];du=recent[-1]['cursor']['updates']-recent[0]['cursor']['updates']
            update_eta=max(0,4000-cursor.get('updates',0))*dt/du if dt>0 and du>0 else None
            if update_eta is not None:
                d['estimated_segment_remaining_seconds']=min(d['estimated_segment_remaining_seconds'] or update_eta,update_eta)
    d['development_pair']=read(LINE/'closed_loop_bench/ordinary_expanded_dev_pair_v1/STATUS.json')
    return d


def html():
    text=r.html().decode()
    old='已接入修复后的续训，地址不变。旧曲线保留；恢复处分段，不跨预算预留跳变计算速度。计费量含保守预留，不是独立样本数。'
    new='扩量后普通导航首段：从 51,301 步参数暖启动，新阶段从 0 计步；最多 4,000 次新更新。265 万动作池，普通人工＋EnvDrop，不含特殊数据。下方是训练指标，不是导航成功率。'
    assert text.count(old)==1;text=text.replace(old,new)
    text=text.replace('总计费预算（含旧失败尾部保守预留）','本阶段动作上限（旧训练另计）')
    text=text.replace('三卡确定性计划计数，含多个epoch；非独立样本','本阶段三卡实际计划计数；不是独立路线')
    marker='</html>'
    banner='''<section style="padding:16px;margin:16px;border:1px solid #50637d;border-radius:12px"><b>数据审核与闭环检查</b><p id="expanded-extra">正在读取；完整基准参照是旧 41,800 步，不是本轮结果。</p></section>
<script>async function expandedStatus(){try{const r=await fetch(new URL('api/status',document.baseURI),{cache:'no-store',signal:AbortSignal.timeout(8000)});if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();const a=d.data_preparation||{};const e=d.development_pair||{};document.getElementById('expanded-extra').textContent='数据审核：'+(a.status||'未知')+'，'+(a.audited??0)+' / '+(a.total??19566)+' 条 EnvDrop；小闭环：'+(e.status||'尚未启动')+'。旧完整验证：SR 22.02%，SPL 18.68%（41,800 步 / 1,839 条）；新模型结果未测不能填零。';}catch(e){document.getElementById('expanded-extra').textContent='附加状态暂时不可达：'+String(e)+'；请检查 SSH 转发，页面将自动重试。';}finally{setTimeout(expandedStatus,10000)}}expandedStatus();</script>'''
    assert text.count(marker)==1
    return text.replace(marker,banner+marker).encode()


r.b.collect=collect
class Handler(r.Handler):
    def do_GET(self):
        if self.path=='/':self.respond(200,html(),'text/html; charset=utf-8')
        else:super().do_GET()
    do_HEAD=do_GET


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);p.add_argument('--check',action='store_true');args=p.parse_args()
    html()
    if args.check:
        d=collect(False);print(json.dumps({k:d[k] for k in ('monitor_version','display_state','decisions','data_preparation')},ensure_ascii=False))
    else:
        with r.b.ThreadingHTTPServer(('127.0.0.1',args.port),Handler) as server:
            server.daemon_threads=True;server.serve_forever(poll_interval=.5)
