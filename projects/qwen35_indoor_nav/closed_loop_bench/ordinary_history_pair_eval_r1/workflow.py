"""Bounded read-only acceptance watcher and one fixed evaluation per accepted arm."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_history8_paired_train_r1'
REVIEW=LINE/'reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
QPY=LINE/'.envs/q35n_qwen_g2_v1/bin/python3'
s=importlib.util.spec_from_file_location('history_workflow_owner',LINE/'sft_acceptance/ordinary_prefix_history8_v1/owned_process_r1.py')
own=importlib.util.module_from_spec(s);s.loader.exec_module(own)
def read(path):
    if not path.exists():return None
    if time.time()-path.stat().st_mtime<2:return None
    return json.loads(path.read_text())
def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:json.dump(obj,stream,indent=2,ensure_ascii=False,allow_nan=False)
    temp.replace(path)
def gpu_empty():
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    gpu=next(g for g in root.findall('gpu') if g.findtext('uuid')=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8')
    return not gpu.findall('processes/process_info') and float(gpu.findtext('fb_memory_usage/used').split()[0])<128
def main():
    assert not (HERE/'WORKFLOW_PROCESS.json').exists()
    for name in ('ordinary_history2_dev_r1','ordinary_history8_dev_r1'):
        assert (HERE.parent/name/'PENDING_SEAL.json').is_file()
    value=own.inspect(os.getpid());assert value
    with (HERE/'WORKFLOW_PROCESS.json').open('x') as stream:json.dump(own.stable(value),stream,indent=2)
    began=time.monotonic();deadline=began+52000;steps=[];stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,_:stop.append(s))
    def check():
        assert not stop,'WORKFLOW_SIGNAL'
        assert time.monotonic()<deadline,'WORKFLOW_DEADLINE'
    def status(phase,arm=None):
        save(HERE/'WORKFLOW_STATUS.json',dict(status=phase,arm=arm,unix=time.time(),completed_steps=steps,wall_seconds=time.monotonic()-began))
    def execute(name,argv):
        check();guard=None
        with (HERE/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                guard=own.OwnedChild(child,argv,ROOT)
                save(HERE/(name+'_PROCESS.json'),guard.identity)
                while guard.check()!='exited':check();time.sleep(2)
                assert child.returncode==0,'STEP_EXIT:'+name+':'+str(child.returncode)
            finally:
                cleanup=guard.cleanup(timeout=90) if guard else dict(identity_registered=False)
                save(HERE/(name+'_CLEANUP.json'),cleanup)
        steps.append(name)
    try:
        for arm,case in [('control_recent2','ordinary_history2_dev_r1'),('treatment_prefix8','ordinary_history8_dev_r1')]:
            while not read(TRAIN/arm/'ACCEPTANCE.json'):
                check();status('WAIT_FIXED4000_TRAINING_ACCEPTANCE',arm)
                failed=read(TRAIN/arm/'LAUNCH_RESULT.json')
                assert not failed or failed['status']=='COMPLETE','TRAINING_ARM_FAILED:'+arm
                lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
                assert not lease or lease['execute_returned'],'TRAINING_LEASE_FAILED'
                first=REVIEW/(arm+'_FIRST200.json')
                if not first.exists() and read(TRAIN/arm/'run_001/checkpoint_000000200.pt.json'):
                    execute(arm+'_first200',[str(QPY),'-I','-B',str(REVIEW/'first200.py'),arm])
                time.sleep(15)
            if not (REVIEW/(arm+'_FIRST200.json')).exists():
                execute(arm+'_first200',[str(QPY),'-I','-B',str(REVIEW/'first200.py'),arm])
            while not gpu_empty():check();status('WAIT_EMPTY_GPU1_NO_SIGNALS',arm);time.sleep(15)
            status('BINDING_FIXED_FINAL',arm)
            execute(arm+'_bind',[str(PY),'-I','-S','-B',str(HERE/'prepare.py'),'bind',arm])
            status('EVALUATING_FIXED100',arm)
            execute(arm+'_eval',[str(PY),'-I','-S','-B',str(HERE.parent/case/'launch.py')])
        status('COMPARING_COMPLETE_INTERNAL100')
        execute('compare',[str(PY),'-I','-S','-B',str(HERE/'compare.py')])
        status('COMPLETE_INTERNAL_DEV_ONLY')
    except BaseException as exc:
        status('FAILED_NO_AUTOMATIC_RETRY')
        save(HERE/'WORKFLOW_FAILURE.json',dict(unix=time.time(),error=repr(exc),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()

