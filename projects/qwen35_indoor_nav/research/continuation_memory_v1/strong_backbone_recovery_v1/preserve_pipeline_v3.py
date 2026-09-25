"""Independent collect -> matched training -> DEV repair selection -> unseen once."""
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
from transfer_pipeline import placeholder,records
from unseen_pipeline import devices
from preserve_review_v3 import summarize


def main(root):
    lock=(root/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(root);p=u.read(root/'PROTOCOL.json');active=[];leased=False;began=time.time()
    hours=sum(u.read(x)['wall_seconds']/3600 for x in (root/'attempts').glob('*.json') if u.read(x).get('gpu') is not None)
    state=dict(status='RUNNING',phase='RESOURCE_CHECK',started_unix=began,planned_capture=100,training_steps_per_arm=1200)
    def save():
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in active if w['gpu'] is not None),
            workers=[dict(name=w['name'],pid=w['p'].pid,gpu=w['gpu'],output=str(w['output']),
                progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else None) for w in active],
            captured_recorded=len(records(root/'capture','evaluation',False)),captured_sealed=len(records(root/'capture','evaluation',True)))
        u.write(root/'STATUS.json',state)
    def interrupt(signum,frame):raise InterruptedError('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def finish(w):
        nonlocal hours
        seconds=time.time()-w['start'];hours+=seconds/3600 if w['gpu'] is not None else 0
        w['log'].close();u.write(w['receipt'],dict(name=w['name'],gpu=w['gpu'],uuid=w['uuid'],pid=w['p'].pid,
            command=w['command'],output=str(w['output']),returncode=w['p'].returncode,wall_seconds=seconds))
    def jobs(items,phase):
        state['phase']=phase;save()
        for name,script,args,gpu,out in items:
            u.verify_sources(root);inventory=devices();uuid=inventory[gpu]['uuid'] if gpu is not None else ''
            if gpu is not None and inventory[gpu]['free_mib']<(26000 if 'worker' in script else 4000):
                raise RuntimeError('GPU_MEMORY_UNAVAILABLE:'+str(gpu))
            logfile=root/'attempts'/(name+'_'+str(time.time_ns())+'.log');logfile.parent.mkdir(exist_ok=True)
            log=logfile.open('x');command=[str(u.PYTHON),'-I','-B',str(u.HERE/script),*map(str,args)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            process=subprocess.Popen(command,env=env,cwd=u.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(dict(name=name,p=process,gpu=gpu,uuid=uuid,output=out,log=log,receipt=logfile.with_suffix('.json'),command=command,start=time.time()))
        while active:
            for w in list(active):
                if w['p'].poll() is not None:
                    finish(w);active.remove(w)
                    if w['p'].returncode:raise RuntimeError('WORKER_FAILED:'+w['name']+':'+str(w['receipt']))
            save()
            if state['gpu_hours']>p['gpu_session_hours_limit'] or time.time()-began>p['wall_hours_limit']*3600:
                raise RuntimeError('REGISTERED_RESOURCE_BOUND')
            if active:time.sleep(5)
    def evaluation(run,ids,phase,capture=False):
        pending=[i for i in ids if i not in records(run,'evaluation',True)]
        # A session without its state seal stays a failed attempt, never a partial causal comparison.
        for folder in (run/'evaluation').glob('*'):
            if not (folder/'STATE_SEAL.json').exists():
                archive=run/'failed_attempts';archive.mkdir(exist_ok=True);folder.rename(archive/folder.name)
        free=[g for g in p['gpu_indices'] if devices()[g]['free_mib']>=26000]
        if pending and not free:raise RuntimeError('NO_AUTHORIZED_FREE_GPU')
        items=[]
        for rank,gpu in enumerate(free):
            shard=pending[rank::len(free)]
            if not shard:continue
            out=run/'evaluation'/('gpu'+str(gpu)+'_'+str(time.time_ns()))
            items.append((phase+'_gpu'+str(gpu),'capture_worker_v3.py' if capture else 'transfer_worker_v2.py',
                ['--run',run,'--output',out,'--ids',','.join(map(str,shard))],gpu,out))
        jobs(items,phase)
        assert set(records(run,'evaluation',True))==set(ids),'INCOMPLETE_SEALED_DENOMINATOR'
        if not capture:u.write(run/'RESULT.json',summarize(run))
    def bind_eval(run,manifest,trial):
        heads={a:dict(path=str(root/'trials'/str(trial)/'training'/a/'FINAL.pt'),
            sha256=u.sha(root/'trials'/str(trial)/'training'/a/'FINAL.pt')) for a in ('BC','B2','OURS')}
        protocol=dict(p['runtime'],heads=heads,parent_protocol_sha256=u.sha(root/'PROTOCOL.json'),trial=trial)
        if run.exists():
            assert u.read(run/'PROTOCOL.json')==protocol and u.read(run/'DATA_MANIFEST.json')==manifest
        else:
            run.mkdir(parents=True);u.write(run/'PROTOCOL.json',protocol);u.write(run/'DATA_MANIFEST.json',manifest)
            u.write(run/'SOURCE_LOCK.json',u.read(root/'SOURCE_LOCK.json'))
    def report(result):
        u.write(root/'RESULT.json',result)
        lines=[result['status'],'',result['reason'],
            '普通导航 SR 保持、任务 FIT 拟合、历史任务闭环收益分别判断；未运行完整 1839 条。',
            '普通训练来自已暴露的官方 TRAIN 子集；unseen 200 不进入训练或权重选择。',
            '任务监督仍是现有调试资产，不能因此升级旧 training_admission 或宣称泛化。','']
        if 'unseen' in result:
            for arm,row in result['unseen']['arms'].items():lines.append(f"{arm}: {row['successes']}/{row['planned']}，SR {row['sr_full']:.1%}")
            lines.append('任务 FIT 保持不是历史恢复成功；方法是否采用仍须独立任务证据。')
        (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    try:
        assert u.read(Path(p['memory_run'])/'STATUS.json')['status']=='COMPLETE','PREVIOUS_RUN_NOT_FINISHED'
        before=placeholder('status')
        if before['manual_paused'] or before['leases'] or before['external_pids']:raise RuntimeError('PLACEHOLDER_CONTROLLER_BUSY')
        acquired=placeholder('acquire');leased=True
        u.write(root/'RESOURCE_LEASE.json',dict(before=before,acquired=acquired,pid=os.getpid(),scope='Own placeholder release, not exclusive machine reservation'))
        evaluation(root/'capture',list(range(100)),'ORDINARY_NATIVE_CAPTURE',True)
        if not (root/'data/ADMISSION.json').exists():jobs([('data','preserve_data_v3.py',['--root',root],None,root/'data')],'BUILD_SHARED_POOLS')
        if not (root/'PRESERVATION_GRADIENT.json').exists():jobs([('gradient','preserve_review_v3.py',['--root',root,'--phase','gradient'],None,root)],'VERIFY_PRESERVATION_GRADIENT')
        selected=None
        for trial,weight in enumerate(p['preservation_weights']):
            state.update(trial=trial,preservation_weight=weight);folder=root/'trials'/str(trial);training=folder/'training'
            if not (training/'INITIAL_42.pt').exists():
                jobs([('initialize_'+str(trial),'preserve_train_v3.py',['--root',root,'--trial',trial,'--arm','BC','--initialize'],None,training)],'SHARED_INITIALIZATION')
            free=[g for g in p['gpu_indices'] if devices()[g]['free_mib']>=4000]
            if len(free)<3:raise RuntimeError('THREE_TRAINING_GPUS_UNAVAILABLE')
            items=[('train_'+str(trial)+'_'+arm,'preserve_train_v3.py',['--root',root,'--trial',trial,'--arm',arm],gpu,training/arm)
                for gpu,arm in zip(free,('BC','B2','OURS')) if not (training/arm/'FINAL.pt').exists()]
            jobs(items,'MATCHED_PRESERVATION_TRAINING')
            if not (folder/'TRAINING_REVIEW.json').exists():
                jobs([('review_'+str(trial),'preserve_review_v3.py',['--root',root,'--trial',trial,'--phase','training'],None,folder)],'FIT_AND_CACHED_DEV_REVIEW')
            review=u.read(folder/'TRAINING_REVIEW.json')
            if not review['retained_task_arms']:
                u.write(folder/'SELECTION.json',dict(selected=False,reason='No B2/Ours head retains the registered 10/10 task action fit'));continue
            manifest=u.read(root/'DEV_MANIFEST.json');dev=folder/'dev';bind_eval(dev,manifest,trial)
            evaluation(dev,[r['id'] for r in manifest['episodes']],'DEV_PAIRED_EVALUATION')
            measured=u.read(dev/'RESULT.json');native=measured['arms']['NATIVE']['successes']
            eligible=[a for a in review['retained_task_arms'] if measured['arms'][a]['successes']>=native]
            decision=dict(selected=bool(eligible),eligible_arms=eligible,trial=trial,preservation_weight=weight,dev_result=str(dev/'RESULT.json'))
            u.write(folder/'SELECTION.json',decision)
            if eligible:selected=decision;break
        if selected is None:
            report(dict(status='NO_PRESERVING_CANDIDATE_ON_DEV',reason='Registered three repair strengths exhausted; no retained task head also preserved DEV SR. No unseen score selected.'))
            state.update(status='COMPLETE_NO_CANDIDATE',phase='REPAIR_SEARCH_COMPLETE');return
        u.write(root/'FROZEN_SELECTION.json',selected)
        unseen=root/'unseen';manifest=u.read(root/'UNSEEN_MANIFEST.json');bind_eval(unseen,manifest,selected['trial'])
        evaluation(unseen,[r['id'] for r in manifest['episodes']],'UNSEEN_200_ONCE')
        result=u.read(unseen/'RESULT.json');preserved=[a for a in selected['eligible_arms'] if result['paired'][a]['delta_sr']>=0]
        report(dict(status='SR_PRESERVED_ON_EXPOSED_UNSEEN_200' if preserved else 'UNSEEN_SR_NOT_PRESERVED',
            reason='Complete matched comparison; aggregate SR preservation is empirical on this exposed subset only.',
            selected=selected,sr_preserving_arms=preserved,unseen=result,
            recovery_efficacy='NOT_MEASURED',adoption='NOT_AUTOMATIC',gpu_hours=hours))
        state.update(status='COMPLETE',phase='UNSEEN_COMPARISON_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(root/('FAILURE_'+str(time.time_ns())+'.json'),dict(state,traceback=traceback.format_exc()))
        report(dict(status=state['status'],reason=str(error),completed_capture=len(records(root/'capture','evaluation',True)),
            recovery_efficacy='NOT_MEASURED',sr_preservation='NOT_ESTABLISHED'));raise
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
        if leased:u.write(root/'PLACEHOLDER_RELEASE.json',dict(result=placeholder('release'),unix=time.time()))
        state['finished_unix']=time.time();save();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=u.HERE/'preservation_runs'/a.run_id
    if (root/'STARTED.json').exists() and not a.resume:raise ValueError('EXISTING_RUN_REQUIRES_RESUME')
    if (root/'STATUS.json').exists() and u.read(root/'STATUS.json')['status'].startswith('COMPLETE'):raise ValueError('COMPLETED_RUN_READ_ONLY')
    u.write(root/'STARTED.json',dict(resume=a.resume,unix=time.time()));main(root)
