"""Preserve failed V1 and freeze a transport-only restart with identical inputs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_stop_calibration_fit_v1'
REVIEW=LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    assert not (HERE/'SOURCE_LOCK.json').exists() and not (HERE/'run_001').exists()
    fail=read(OLD/'run_001/LAUNCH_RESULT.json');error=read(OLD/'run_001/LAUNCH_FAILURE.json')
    assert fail['status']=='SERVICE_FAILED' and fail['reason']=='LAUNCHER_ERROR'
    assert not fail['cleanup']['remaining'] and not fail['foreign_processes_signaled']
    assert 'FileNotFoundError' in error['traceback'] and '/cache/triton/' in error['traceback']
    assert not list((OLD/'run_001/lanes').glob('lane_*/episode_*.json'))
    assert not any(p.stat().st_size for p in (OLD/'run_001/lanes').glob('lane_*/POLICY_STEPS.jsonl'))
    outputs=[]
    for script in (HERE/'test_tree_size.py',REVIEW/'test_counterfactual.py'):
        r=subprocess.run([str(PY),'-I','-S','-B',str(script)],capture_output=True,text=True,timeout=60)
        assert r.returncode==0,r.stderr;outputs.append(dict(script=str(script),output=r.stdout+r.stderr))
    s=importlib.util.spec_from_file_location('stop_fit_v2_freeze',HERE/'reuse.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
        text=m.source(name);ast.parse(text)
        if name!='launch.py':assert text==m.parent.source(name)
        else:assert 'size,transient_misses=_scan.tree_size(OUT)' in text
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,tests=outputs,scientific_implementation_unchanged=True))
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json','SELECTION.json'):
        with (HERE/name).open('xb') as f:f.write((OLD/name).read_bytes())
        assert sha(HERE/name)==sha(OLD/name)
    p=read(OLD/'PROTOCOL.json');p['id']='Q35N_ORDINARY_STOP_CALIBRATION_FIT_V2'
    save(HERE/'PROTOCOL.json',p)
    save(HERE/'PREDECESSOR_FAILURE.json',dict(path=str(OLD/'run_001'),status=fail['status'],
         failure_sha256=sha(OLD/'run_001/LAUNCH_FAILURE.json'),wall_seconds=fail['wall_seconds'],
         observed_navigation_actions=0,model_load_cost_not_ignored=True,old_failure_preserved=True))
    files=dict(read(OLD/'SOURCE_LOCK.json')['files'])
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(REVIEW.glob('*.py'))+[REVIEW/'TRANSPORT_REVISION_V2.md',OLD/'run_001/LAUNCH_RESULT.json',OLD/'run_001/LAUNCH_FAILURE.json']
    for path in paths:files[str(path)]=sha(path)
    for path,digest in files.items():assert Path(path).resolve().is_relative_to(ROOT) and sha(path)==digest,path
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='single transport-only retry of same 64 FIT routes',training_updates=0))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_BOUNDED_TRANSPORT_RETRY',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
         same_input_sha256=sha(HERE/'EPISODES_PRIVILEGED.json'),gpu=1,requires_empty_gpu=True,
         borrowed_holders_allowed=False,automatic_retry=False,additional_wall_budget_seconds=3600,unix=time.time()))
    print(json.dumps(dict(status='FROZEN',files=len(files),identical_input=True,previous_failure_preserved=True)))


if __name__=='__main__':main()
