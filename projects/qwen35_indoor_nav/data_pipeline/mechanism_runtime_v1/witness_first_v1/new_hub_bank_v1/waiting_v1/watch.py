"""Only observe a parent-owned scout; then run the fixed CPU postprocessor once."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
BANK=HERE.parent
SOURCE=BANK.parent/'new_hub_scout_v1/run_v1'
ROOT=next(p for p in HERE.parents if p.name=='vla')
PYTHON=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
DEADLINE_SECONDS=75*60
INTERVAL_SECONDS=40

def save(path,value):
    assert path.parent==HERE
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def optional(path):
    try:return json.loads(path.read_text()),None
    except FileNotFoundError:return None,None
    except (OSError,json.JSONDecodeError) as exc:return None,repr(exc)
def observe():
    values={};errors={}
    for name in ('PROGRESS.json','result.json','SUPERVISOR_RESULT.json','LAUNCH_RESULT.json'):
        value,error=optional(SOURCE/name)
        if value is not None:values[name]=value
        if error:errors[name]=error
    progress=values.get('PROGRESS.json',{})
    event={'observed_unix':time.time(),'source_exists':SOURCE.exists(),
        'actual_actions':progress.get('actual_actions'),'current_context':progress.get('current_context'),
        'source_result_status':values.get('result.json',{}).get('status'),
        'supervisor_returncode':values.get('SUPERVISOR_RESULT.json',{}).get('returncode'),
        'supervisor_cleanup':values.get('SUPERVISOR_RESULT.json',{}).get('cleanup_complete'),
        'launch_returned':values.get('LAUNCH_RESULT.json',{}).get('supervisor_returned'),
        'read_errors':errors,'gpu_operations':0}
    return values,event
def main():
    owner=(HERE/'WATCHER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'result.json').exists() and not (HERE/'SAMPLES.jsonl').exists(),'NO_DUPLICATE_WATCHER'
    started=time.monotonic();deadline=started+DEADLINE_SECONDS
    save(HERE/'WATCHER_START.json',{'started_unix':time.time(),'max_seconds':DEADLINE_SECONDS,
        'sampling_seconds':INTERVAL_SECONDS,'source':str(SOURCE),'does_not_launch_or_control_scout':True,
        'postprocessor':str(BANK/'prepare.py'),'gpu_operations':0,'scientific_pass':False})
    outcome={'status':'WAITING_TIMEOUT_NO_SOURCE_INTERVENTION','gpu_operations':0,'scientific_pass':False}
    try:
        while time.monotonic()<deadline:
            values,event=observe();event['watch_elapsed_seconds']=time.monotonic()-started
            with (HERE/'SAMPLES.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush()
            print(json.dumps(event),flush=True)
            if all(name in values for name in ('result.json','SUPERVISOR_RESULT.json','LAUNCH_RESULT.json')):
                result=values['result.json'];supervisor=values['SUPERVISOR_RESULT.json'];launch=values['LAUNCH_RESULT.json']
                if not (result['status']=='SCOUT_CLOSED' and result['error'] is None and supervisor['returncode']==0
                        and supervisor['error'] is None and supervisor['cleanup_complete'] is True
                        and launch['supervisor_returned'] is True and launch['error'] is None):
                    outcome.update(status='SOURCE_TERMINAL_NOT_ELIGIBLE_NO_POSTPROCESS',terminal_observation=event);break
                sealed={line.split(None,1)[1]:line.split(None,1)[0] for line in (BANK/'SHA256SUMS').read_text().splitlines()}
                for relative,h in sealed.items():assert hashlib.sha256((BANK/relative).read_bytes()).hexdigest()==h,'BANK_SOURCE_CHANGED'
                for phase,args in [('source_check',['--check-only']),('cpu_prepare',[])]:
                    remaining=deadline-time.monotonic();assert remaining>0,'WATCH_TOTAL_DEADLINE'
                    with (HERE/(phase+'.log')).open('x') as log:
                        proc=subprocess.run([str(PYTHON),'-I','-S','-B',str(BANK/'prepare.py'),*args],
                            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                    outcome[phase+'_returncode']=proc.returncode
                    if proc.returncode!=0:outcome['status']=phase.upper()+'_FAILED_OUTPUTS_PRESERVED';break
                else:outcome['status']='CPU_BANK_PREPARED_AFTER_NATURAL_SCOUT_CLOSE'
                break
            remaining=deadline-time.monotonic()
            if remaining>0:time.sleep(min(INTERVAL_SECONDS,remaining))
    except BaseException as exc:
        outcome.update(status='WATCH_OR_CPU_POSTPROCESS_ERROR_OUTPUTS_PRESERVED',error=repr(exc));raise
    finally:
        outcome['elapsed_seconds']=time.monotonic()-started;save(HERE/'result.json',outcome);print(json.dumps(outcome),flush=True)
if __name__=='__main__':main()
