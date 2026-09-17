"""Read-only parent-owned batch08 monitoring, then isolated strong CPU audit."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
GATE=HERE.parent/'gate.py'
GATE_SHA='9aa097b918611537a9d0f2517528a6e257ec6504e860cc53fe28af8c9119fb83'
WF=HERE.parents[3]
RUN=WF/'batch_execution_v1/batch_08/run_v1'
PROJECT=next(p for p in HERE.parents if p.name=='vla')
PYTHON=PROJECT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'

def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(PROJECT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):
    assert path.is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def optional(path):
    try:return read(path),None
    except FileNotFoundError:return None,None
    except (OSError,json.JSONDecodeError) as exc:return None,repr(exc)
def observed():
    files={};errors={}
    for name in ('PROGRESS.json','result.json','SUPERVISOR_RESULT.json','LAUNCH_RESULT.json'):
        value,error=optional(RUN/name)
        if value is not None:files[name]=value
        if error:errors[name]=error
    progress=files.get('PROGRESS.json',{})
    row={'observed_unix':time.time(),'actual_actions':progress.get('actual_actions'),
        'traces':progress.get('traces'),'complete_traces':progress.get('complete_traces'),
        'worker_reported_physical_families_not_audited':progress.get('physical_families'),
        'collisions':progress.get('collisions'),'current_bundle':progress.get('current_bundle'),
        'supervisor_returncode':files.get('SUPERVISOR_RESULT.json',{}).get('returncode'),
        'cleanup_complete':files.get('SUPERVISOR_RESULT.json',{}).get('cleanup_complete'),
        'launch_status':files.get('LAUNCH_RESULT.json',{}).get('status'),'read_errors':errors,'gpu_operations':0}
    return files,row
def audit():
    assert sha(GATE)==GATE_SHA,'FROZEN_GATE_ADAPTER_CHANGED'
    terminal=read(RUN/'SUPERVISOR_RESULT.json');read(RUN/'LAUNCH_RESULT.json')
    # Even terminal failures may be audited, but never bypass original checks.
    assert isinstance(terminal,dict)
    started=read(HERE/'WATCH_START.json')
    for path,h in started['frozen_inputs'].items():assert sha(Path(path))==h,path
    s=importlib.util.spec_from_file_location('batch08_independent_strong_gate3',GATE)
    g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
    cfg=read(RUN/'EXECUTION_CONFIG.json')
    assert cfg['source_snapshot']==str(WF/'new_hub_bank_v1/language_ready_v1')
    assert cfg['source_selection_indices']==[0,24,48] and len(cfg['candidates'])==3
    out=HERE/'audit_v1';out.mkdir(exist_ok=False)
    # Same observe_run -> inspect_run -> build_family_evidence -> quality chain.
    # No calls to all-history cohort aggregation, recovery upgrades or GPU APIs.
    observation=g.gate.observe_run(RUN,out)
    passed=[r for r in observation['attempts'] if r.get('quality_pass') is True and r.get('source_and_phase_binding_verified') is True]
    report={'status':'BATCH08_INDEPENDENT_STRONG_AUDIT_COMPLETE','run_root':str(RUN),'batch_observation':observation,
        'registered_attempt_count':3,'strongly_accepted_family_count':len(passed),
        'all_three_registered_attempts_strongly_accepted':len(passed)==3 and observation['all_frozen_candidates_attempted_and_terminal'],
        'all_attempts_included':True,'fit_houses':len({r['house_id'] for r in cfg['candidates']}),
        'registered_physical_hubs':len(g.gate.old.independent_hubs([{'house_id':r['house_id'],'hub_position':r['configuration']['u_position']} for r in cfg['candidates']])),
        'scope':'same_house_new_hub_data_generation_only','same_house_hubs_correlated':True,
        'added_to_old_cohort':False,'cohort_pass':False,'statistical_stability_pass':False,
        'model_generalization_pass':False,'algorithm_gain_claim':False,'gpu_operations':0,'scientific_pass':False,
        'gate_source_sha256':GATE_SHA}
    save(HERE/'REPORT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='batch_observation'},indent=2))
def watch():
    lock=(HERE/'WATCHER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'WATCH_START.json').exists(),'ONE_BOUNDED_WATCH_ONLY'
    assert sha(GATE)==GATE_SHA
    save(HERE/'WATCH_START.json',{'started_unix':time.time(),'watch_seconds':4500,'poll_seconds':40,'cpu_audit_timeout_seconds':900,
        'run_root':str(RUN),'frozen_inputs':{str(p):sha(p) for p in (GATE,RUN/'EXECUTION_CONFIG.json',RUN/'INPUT_LOCK.json',HERE/'watch_and_audit.py')},
        'old_cohort_untouched':True,'gpu_operations':0,'scientific_pass':False})
    started=time.monotonic();deadline=started+4500;outcome={'status':'WAIT_TIMEOUT_SOURCE_UNTOUCHED','gpu_operations':0,'scientific_pass':False}
    try:
        while time.monotonic()<deadline:
            files,row=observed();row['watch_elapsed_seconds']=time.monotonic()-started
            with (HERE/'SAMPLES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps(row),flush=True)
            if 'SUPERVISOR_RESULT.json' in files and 'LAUNCH_RESULT.json' in files:
                with (HERE/'audit.log').open('x') as log:
                    proc=subprocess.run([str(PYTHON),'-I','-S','-B',str(HERE/'watch_and_audit.py'),'--audit'],
                        cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                outcome.update(status='INDEPENDENT_AUDIT_FINISHED' if proc.returncode==0 else 'AUDIT_FAILED_OUTPUTS_PRESERVED',cpu_audit_returncode=proc.returncode);break
            time.sleep(min(40,max(0,deadline-time.monotonic())))
    except BaseException as exc:outcome.update(status='WATCH_OR_AUDIT_ERROR_OUTPUTS_PRESERVED',error=repr(exc));raise
    finally:
        outcome['elapsed_seconds']=time.monotonic()-started;save(HERE/'result.json',outcome);print(json.dumps(outcome),flush=True)
if __name__=='__main__':
    if sys.argv[1:]==['--audit']:audit()
    else:
        assert not sys.argv[1:];watch()
