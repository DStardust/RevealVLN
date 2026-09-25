"""Independent bounded real-feature -> matched-training -> diagnostic job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def main(run_id):
    run=u.HERE/'method_runs'/run_id;run.mkdir(parents=True,exist_ok=False)
    data=u.HERE/'data_runs/moving_003';source=u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')
    families=[f['family_id'] for f in source['families']];began=time.time();children=[];gpu_seconds=0.;active=[]
    names=['method_pipeline.py','extract_history_features.py','build_training_pack.py','memory_v2.py','train_memory.py','review_memory.py','common.py']
    locked={str(u.HERE/name):u.sha(u.HERE/name) for name in names}
    u.write(run/'SOURCE_LOCK.json',locked)
    protocol=dict(id=run_id,data=str(data),families=families,arms=['BC','B2','OURS'],seed=42,steps_per_arm=1200,
        max_wall_seconds=14400,max_gpu_hours=12,exposed_two_house_debug_only=True,base_updates=0,
        memory='Shared existing 8x64 recurrence; extra frozen visual observation every actual step, task + executed previous action conditioning',
        actor='Residual on native action-token logits; action model keeps four-step generation cadence',
        scope='Real feature extraction, matched full-sequence updates, fixed action diagnostic; closed-loop method gain remains unmeasured',
        training_shared='Same initialization, dataset, teacher, shuffled schedule and optimizer budget',
        changed_from_first_zero_adapter='Continuous visual writer inputs added equally to BC/B2/OURS to cover physical SEE2 witnesses; old native result is not the method control',
        no_automatic_retries=True)
    u.write(run/'PROTOCOL.json',protocol);u.write(u.HERE/'METHOD_RUN.json',dict(run=run_id,path=str(run)))
    def status(phase,**extra):
        elapsed=gpu_seconds+sum(time.time()-p['started'] for p in active)
        u.write(run/'STATUS.json',dict(status='RUNNING',phase=phase,wall_seconds=time.time()-began,gpu_hours=elapsed/3600,
            active=[{k:v for k,v in p.items() if k not in ('proc','handle')} for p in active],**extra))
        if time.time()-began>protocol['max_wall_seconds'] or elapsed>protocol['max_gpu_hours']*3600:raise RuntimeError('RESOURCE_BUDGET')
    def launch(name,script,args,gpu):
        for path,digest in locked.items():
            if u.sha(path)!=digest:raise RuntimeError('SOURCE_CHANGED_DURING_RUN')
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        line=next(line for line in devices.splitlines() if int(line.split(',')[0])==gpu);index,uuid,free=[x.strip() for x in line.split(',')]
        if int(free)<(26000 if script=='extract_history_features.py' else 4000):raise RuntimeError('GPU_RESOURCE_NOT_AVAILABLE:'+index)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        handle=(run/(name+'.log')).open('x');command=[str(u.PYTHON),'-B',str(u.HERE/script),*map(str,args)]
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=name,pid=proc.pid,gpu=gpu,gpu_uuid=uuid,started=time.time(),command=command,proc=proc,handle=handle)
        active.append(row);children.append(row);return row
    def reap(phase):
        nonlocal gpu_seconds
        while active:
            status(phase)
            for row in list(active):
                code=row['proc'].poll()
                if code is not None:
                    active.remove(row);gpu_seconds+=time.time()-row['started'];row['handle'].close()
                    u.write(run/(row['name']+'.EXIT.json'),dict(returncode=code,pid=row['pid'],gpu=row['gpu'],wall_seconds=time.time()-row['started']))
                    if code:raise RuntimeError('WORKER_FAILED:'+row['name'])
            if active:time.sleep(5)
    try:
        while not (data/'RESULT.json').exists():
            status('WAITING_FOR_PHYSICAL_REPLAY')
            if (data/'STATUS.json').exists() and u.read(data/'STATUS.json')['status']=='FAILED':raise RuntimeError('PHYSICAL_REPLAY_FAILED')
            time.sleep(5)
        assert u.read(data/'STATUS.json')['completed_traces']==72
        feature=run/'features';feature.mkdir()
        # The first complete family exercises the actual cache path before
        # occupying all six cards. No score is used to admit the remaining five.
        launch('feature_0','extract_history_features.py',['--data',data,'--output',feature/families[0],'--family',families[0]],2);reap('FEATURE_PILOT')
        for index,fid in enumerate(families[1:],1):
            launch('feature_'+str(index),'extract_history_features.py',['--data',data,'--output',feature/fid,'--family',fid],index+2)
        reap('FEATURES')
        pack=run/'data'
        launch('build_pack','build_training_pack.py',['--data',data,'--features',feature,'--output',pack],2);reap('BUILD_PACK')
        training=run/'training'
        # Initialize once before parallel arms: same initial bytes and schedule.
        launch('initialize','train_memory.py',['--data',pack/'FIT.pt','--run',training,'--arm','BC','--initialize-only'],2);reap('INITIALIZE')
        for gpu,arm in zip((2,3,4),protocol['arms']):
            launch('train_'+arm,'train_memory.py',['--data',pack/'FIT.pt','--run',training,'--arm',arm],gpu)
        reap('TRAINING')
        launch('review','review_memory.py',['--data',pack,'--training',training,'--output',run/'ACTION_REVIEW.json'],2);reap('REVIEW')
        u.write(run/'STATUS.json',dict(status='COMPLETE',phase='ACTION_DIAGNOSTIC_COMPLETE',wall_seconds=time.time()-began,
            gpu_hours=gpu_seconds/3600,actual_updates=3600,base_updates=0,closed_loop_method_benefit='NOT_YET_MEASURED'))
    except BaseException as error:
        u.write(run/'STATUS.json',dict(status='FAILED',error=repr(error),wall_seconds=time.time()-began,
            gpu_hours=(gpu_seconds+sum(time.time()-p['started'] for p in active))/3600,closed_loop_method_benefit='UNKNOWN'))
        raise
    finally:
        # Only direct child process groups created by this invocation.
        for row in children:
            p=row['proc']
            if p.poll() is None and os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
        for row in children:
            try:row['proc'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.getpgid(row['pid'])==row['pid']:os.killpg(row['pid'],signal.SIGKILL)
            row['handle'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(a.run_id)
