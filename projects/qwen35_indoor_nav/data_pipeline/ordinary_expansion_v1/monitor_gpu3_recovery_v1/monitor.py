"""Read-only aggregation of old GPU3 closure and the disjoint recovery namespace."""
import copy
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'monitor_recovery_v1'
RECOVERY=HERE.parent/'runtime_gpu3_recovery_v1'
DATA=HERE.parent/'envdrop_production_gpu3_recovery_v1'
raw=(PREVIOUS/'monitor.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='940bfa605b826639e7d7f19ff211684a109ea21106462196f47e196e747be332'
spec=importlib.util.spec_from_file_location('prior_readonly_monitor',PREVIOUS/'monitor.py')
prior=importlib.util.module_from_spec(spec);exec(compile(raw,str(PREVIOUS/'monitor.py'),'exec'),prior.__dict__)
VERSION='expansion-monitor-gpu3-recovery-v1'
prior.VERSION=VERSION
ROOT=prior.ROOT
LINE=prior.LINE
OLD=prior.OLD
sha=prior.sha
read=prior.read
SnapshotCache=prior.SnapshotCache
Server=prior.Server
handler_for=prior.handler_for
base_loader=prior.load_collector

def augment(base,parts,salvage,gate,final,launched):
    data=copy.deepcopy(base)
    lane=next(l for l in data['lanes'] if l['gpu']==3)
    legacy=copy.deepcopy(lane)
    lane['prior_failure']=legacy['result']
    old_strict=salvage.get('strict_routes',legacy['strict_routes'])
    old_decisions=salvage.get('instruction_conditioned_decisions',legacy['strict_decisions'])
    assert legacy['completed']==140 and old_strict in (103,140),'GPU3_OLD_CLOSURE_CHANGED'
    lane['completed']=legacy['completed']+sum(p.get('completed',0) for p in parts)
    lane['replay_certified']=legacy['replay_certified']+sum(p.get('replay_certified_routes',0) for p in parts)
    lane['strict_routes']=old_strict+sum(p.get('audited_routes',0) for p in parts)
    lane['strict_decisions']=old_decisions+sum(p.get('audited_instruction_conditioned_decisions',0) for p in parts)
    lane['result']=final
    lane['sentinel_pass']=gate.get('strict_pass')
    lane['stage']=('FAILED' if final.get('error') else 'CLOSED') if final else ('PRODUCING' if gate.get('strict_pass') else 'FIRST_STRICT_GATE')
    if not launched:lane['stage']='RECOVERY_PREPARED'
    lane['recovery']=dict(version='runtime_gpu3_recovery_v1',assigned=3112,
        old_terminal=140,partial_not_retried=1,current_shard=parts[-1]['shard'] if parts else None,
        completed_new=sum(p.get('completed',0) for p in parts),old_salvage=salvage,
        old_failure_retained=True,automatic_training_restart=False)
    lane['latest_shard']=parts[-1]['shard'] if parts else None
    lane['heartbeat_unix']=parts[-1]['heartbeat_unix'] if parts else None
    for key in ('completed','strict_routes','strict_decisions'):
        data[key]=sum(l[key] for l in data['lanes'])
    assert lane['completed']<=3252 and data['completed']<=9660,'RECOVERY_DUPLICATE_COUNT'
    data['note']+=' GPU3 已登记独立恢复队列：原 140 条完整样本保留，旧超时失败可追溯，1 条中断路线不重试。总计划 9661 包含这 1 条未完成；恢复数据不与旧计数重复。'
    data['gpu3_recovery_registered']=True
    return data

def load_collector():
    collect_base=base_loader()
    def collect():
        base=collect_base()
        if not (RECOVERY/'INPUT_LOCK.json').exists():return base
        cfg=read(RECOVERY/'PREPARED_CONFIG.json')
        assert cfg['routes']==3112 and cfg['gpu_to_shards']=={'3':list(range(64))}
        parts=[]
        for s in range(64):
            path=DATA/f'shard_{s:04d}/PROGRESS.json';p=read(path)
            if p:parts.append(dict(p,shard=s,heartbeat_unix=path.stat().st_mtime))
        out=RECOVERY/'lanes/gpu_3/attempt_000'
        return augment(base,parts,read(RECOVERY/'old_salvage/RESULT.json'),read(out/'SENTINEL_GATE.json'),
                       read(out/'RESULT.json'),(RECOVERY/'launch_001/production_PROCESS.json').exists())
    return collect

prior.load_collector=load_collector

if __name__=='__main__':prior.main()
