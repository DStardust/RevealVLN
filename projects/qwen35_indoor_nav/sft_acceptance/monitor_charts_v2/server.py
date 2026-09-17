"""Read-only chart extension: recent action quality and approved queue drains."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'monitor_charts_v1'
SPEED=HERE.parent/'ordinary_speedup_10x_v1'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

for name,digest in json.loads((BASE/'CODE_SEAL.json').read_text()).items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,'FROZEN_MONITOR_CHANGED'
b=load('sealed_charts_v1',BASE/'server.py')
q=load('recent_quality_diagnostics',SPEED/'quality_diagnostics.py')
original_collect=b.collect

def collect(*args,**kwargs):
    d=original_collect(*args,**kwargs)
    records=b.tail_records(b.EXECUTION/d['run_name']/'PROGRESS.jsonl')['records']
    d['recent_quality']=q.summarize(records,1000)
    for row in d['queues']:
        gpu=row.get('gpu')
        if gpu not in (3,4,5) or row.get('plan')!=f'special_holder_gpu{gpu}_v1':continue
        receipt=b.read_json(SPEED/f'DRAIN_REQUEST_GPU{gpu}.json')['data'] or {}
        identity=receipt.get('exact_queue') or {}
        if row.get('alive') and row.get('pid')==identity.get('pid'):
            row['state']='DRAINING_CURRENT_BATCH_AND_AUDIT'
            row['note']='用户已允许让卡；当前批次/清理/强审完成后不接续，尚不可当空闲卡使用'
    return d

b.collect=collect
b.HERE=HERE
if __name__=='__main__':
    for name,digest in json.loads((HERE/'CODE_SEAL.json').read_text()).items():
        path=HERE/name
        assert path.resolve().is_relative_to(b.LINE) and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    b.main()
