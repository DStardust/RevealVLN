"""Action-position repair: live probe, corrected caches, matched heads, evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
from unseen_pipeline import devices
from transfer_pipeline import placeholder, records, summarize


def main(root,phase):
    lock=(root/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(root);protocol=u.read(root/'PROTOCOL.json')
    started=time.time();active=[];leased=False
    hours=sum(u.read(p)['wall_seconds']/3600 for p in (root/'attempts').glob('*.json') if u.read(p).get('gpu') is not None)
    state=dict(status='RUNNING',phase='PROBE',started_unix=started,training_steps_per_arm=1200)
    def progress():
        return [dict(name=w['name'],pid=w['p'].pid,gpu=w['gpu'],output=str(w['output']),
            progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else
                u.read(w['output']/'STATUS.json') if w['output']!=root and (w['output']/'STATUS.json').exists() else None) for w in active]
    def save():
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in active if w['gpu'] is not None),workers=progress())
        ordinary=root/'ordinary'
        if ordinary.exists():
            state.update(recorded_groups=len({p.parent.name for p in ordinary.glob('evaluation/*/episodes/*/COMPLETE.json')}),
                sealed_groups=len(records(ordinary,'evaluation',True)))
            u.write(ordinary/'LIVE_RESULT.json',summarize(ordinary))
        u.write(root/'STATUS.json',state)
    def interrupt(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def finish(w):
        nonlocal hours
        duration=time.time()-w['start']
        if w['gpu'] is not None:hours+=duration/3600
        w['log'].close()
        u.write(w['receipt'],dict(name=w['name'],gpu=w['gpu'],uuid=w['uuid'],pid=w['p'].pid,
            command=w['command'],output=str(w['output']),returncode=w['p'].returncode,wall_seconds=duration))
    def run_jobs(jobs,stage):
        state['phase']=stage;inventory=devices();save()
        for name,script,args,gpu,output in jobs:
            u.verify_sources(root)
            if gpu is not None and inventory[gpu]['free_mib']<(26000 if script in ('transfer_worker_v2.py','extract_history_features_v2.py') else 4000):
                raise RuntimeError('GPU_RESOURCE_UNAVAILABLE:'+str(gpu))
            uuid=inventory[gpu]['uuid'] if gpu is not None else ''
            attempt=name+'_'+str(time.time_ns());logpath=root/'attempts'/(attempt+'.log');logpath.parent.mkdir(exist_ok=True)
            log=logpath.open('x');command=[str(u.PYTHON),'-I','-B',str(u.HERE/script),*map(str,args)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            p=subprocess.Popen(command,env=env,cwd=u.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(name=name,p=p,gpu=gpu,uuid=uuid,output=output,log=log,receipt=logpath.with_suffix('.json'),command=command,start=time.time()))
        while active:
            for w in list(active):
                if w['p'].poll() is not None:
                    finish(w);active.remove(w)
                    if w['p'].returncode:raise RuntimeError('WORKER_FAILED:'+w['name']+':'+str(w['receipt']))
            save()
            if state['gpu_hours']>protocol['gpu_session_hours_limit'] or time.time()-started>protocol['wall_hours_limit']*3600:
                raise RuntimeError('REGISTERED_RESOURCE_BUDGET')
            if active:time.sleep(5)
    try:
        probe=root/'probe'
        complete=list(probe.glob('sessions/*/RESULT.json'))
        if not complete:
            output=probe/'sessions'/str(time.time_ns())
            run_jobs([('live_probe','transfer_worker_v2.py',['--run',probe,'--output',output,'--ids','0,1','--identity-pairs'],0,output)],'LIVE_ACTION_PROBE')
            assert u.read(output/'RESULT.json')['base_unchanged']
            assert u.read(output/'episodes/0/NONZERO_PROBE/PROBE_EXECUTION.json')['status']=='NONZERO_ACTION_EXECUTED'
        state['live_probe']='ZERO_IDENTITY_AND_NONZERO_ENV_EXECUTION_VERIFIED'
        if phase=='probe':
            state.update(status='PROBE_COMPLETE',phase='LIVE_ACTION_PROBE_COMPLETE');save();return
        before=placeholder('status')
        if before['manual_paused'] or before['leases'] or before['external_pids']:
            raise RuntimeError('PLACEHOLDER_UNAVAILABLE:'+str(before))
        leased=True;u.write(root/'PLACEHOLDER_LEASE.json',dict(before=before,acquired=placeholder('acquire'),pid=os.getpid()))
        features=root/'features';features.mkdir(exist_ok=True)
        index_path=features/'FEATURE_INDEX.json';index=u.read(index_path) if index_path.exists() else {}
        pending=[];jobs=[]
        for rank,fid in enumerate(protocol['families']):
            if fid in index:
                assert u.read(Path(index[fid])/'RESULT.json')['action_boundary']=='AFTER_NATIVE_ASSISTANT_HEADER_V2'
                continue
            out=features/(fid+'_'+str(time.time_ns()));pending.append((fid,out))
            jobs.append(('features_'+fid,'extract_history_features_v2.py',['--data',protocol['data'],'--output',out,'--family',fid],rank+2,out))
        # Finished family caches are reusable after a resource interruption.
        try:run_jobs(jobs,'CORRECTED_FEATURES')
        finally:
            for fid,out in pending:
                if (out/'RESULT.json').exists():index[fid]=str(out)
            u.write(index_path,index)
        data=root/'data'
        if not (data/'ADMISSION.json').exists():
            out=data if not data.exists() else root/('data_attempt_'+str(time.time_ns()))
            run_jobs([('build_data','build_training_pack_v2.py',['--data',protocol['data'],'--features',features,'--output',out],None,out)],'DATA_BINDING')
            if out!=data:raise RuntimeError('PARTIAL_DATA_DIRECTORY_REQUIRES_EXPLICIT_INDEX_REPAIR')
        training=root/'training'
        if not (training/'INITIAL_42.pt').exists():
            run_jobs([('initialize','train_memory_v2.py',['--data',data/'FIT.pt','--run',training,'--arm','BC','--initialize-only','--device','cpu'],None,training/'BC')],'SHARED_INITIALIZATION')
        jobs=[]
        for gpu,arm in zip((2,3,4),('BC','B2','OURS')):
            if (training/arm/'FINAL.pt').exists():
                assert u.read(training/arm/'RESULT.json')['updates']==1200;continue
            jobs.append(('train_'+arm,'train_memory_v2.py',['--data',data/'FIT.pt','--run',training,'--arm',arm],gpu,training/arm))
        run_jobs(jobs,'MATCHED_TRAINING')
        run_jobs([('training_review','review_corrected_v2.py',['--root',root,'--phase','training'],None,root)],'TRAINING_REVIEW')
        ordinary=root/'ordinary'
        if not (ordinary/'PROTOCOL.json').exists():
            ordinary.mkdir(exist_ok=True)
            previous=u.read(u.HERE/'transfer_runs/ordinary_001/PROTOCOL.json')
            previous.update(id='STREAMVLN_ACTION_POSITION_V2',heads={a:dict(path=str(training/a/'FINAL.pt'),sha256=u.sha(training/a/'FINAL.pt')) for a in ('BC','B2','OURS')},
                action_rule='Learned residual at FIRST REAL ACTION after naturally generated assistant header; subsequent decoding unchanged',
                training_cache='Corrected header-preserving teacher forcing and real action-position features',parent_repair_protocol=str(root/'PROTOCOL.json'))
            previous.pop('golden',None)
            u.write(ordinary/'PROTOCOL.json',previous)
            u.write(ordinary/'DATA_MANIFEST.json',u.read(u.HERE/'runs/unseen_001/DATA_MANIFEST.json'))
            files=dict(u.read(root/'SOURCE_LOCK.json')['files'])
            for name in ('PROTOCOL.json','DATA_MANIFEST.json'):files[str(ordinary/name)]=u.sha(ordinary/name)
            u.write(ordinary/'SOURCE_LOCK.json',dict(files=files))
        u.write(ordinary/'STATUS.json',dict(status='RUNNING'))
        for label,ids in [('EVALUATION_FIRST_FIVE',list(range(5))),('EVALUATION_REMAINING',list(range(5,200)))]:
            done=records(ordinary,'evaluation',True);pending=[i for i in ids if i not in done]
            gpus=[0] if label=='EVALUATION_FIRST_FIVE' else list(range(8));jobs=[]
            for rank,gpu in enumerate(gpus):
                shard=pending[rank::len(gpus)]
                if not shard:continue
                out=ordinary/'evaluation'/f'gpu{gpu}_{time.time_ns()}'
                jobs.append((label.lower()+'_gpu'+str(gpu),'transfer_worker_v2.py',['--run',ordinary,'--output',out,'--ids',','.join(map(str,shard))],gpu,out))
            run_jobs(jobs,label)
        result=summarize(ordinary);assert result['complete_groups']==200
        result.update(status='COMPLETE',gpu_hours=state['gpu_hours'])
        u.write(ordinary/'RESULT.json',result);u.write(ordinary/'STATUS.json',dict(status='COMPLETE',gpu_hours=state['gpu_hours']))
        run_jobs([('navigation_review','review_corrected_v2.py',['--root',root,'--phase','navigation'],None,root)],'FINAL_REVIEW')
        state.update(status='COMPLETE',phase='MATCHED_COMPARISON_COMPLETE',actual_optimizer_steps=3600)
    except BaseException as error:
        state.update(status='FAILED',error=str(error))
        u.write(root/f'FAILURE_{time.time_ns()}.json',dict(state,traceback=traceback.format_exc()))
        raise
    finally:
        for w in active:
            if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGTERM)
        for w in active:
            try:w['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGKILL)
                w['p'].wait(timeout=10)
            finish(w)
        active.clear()
        if leased:
            try:u.write(root/'PLACEHOLDER_RELEASE.json',dict(result=placeholder('release'),unix=time.time()))
            except Exception as error:state['placeholder_release_error']=str(error)
        state['finished_unix']=time.time();save();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['probe','all'],default='all');a=p.parse_args()
    main(u.HERE/'repair_runs'/a.run_id,a.phase)

