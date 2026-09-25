"""Independent collection -> TRAIN fitting -> real unseen evaluation, resumable."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
from transfer_pipeline import placeholder,records
from unseen_pipeline import devices
from prepare_recovery import prepare_unseen


def summarize(run):
    protocol=u.read(run/'PROTOCOL.json');manifest={e['id']:e for e in u.read(run/'DATA_MANIFEST.json')['episodes']}
    rows=records(run,'evaluation',True);n=len(manifest);arms={};paired={};houses={}
    for arm in ['NATIVE']+list(protocol['heads']):
        values=[r['outcomes'][arm] for r in rows.values()];p=sum(v['success'] for v in values)
        arms[arm]=dict(complete=len(values),planned=n,successes=p,sr=p/n if len(values)==n else None,
            identification_bounds=[p/n,(p+n-len(values))/n],spl=sum(v['spl'] for v in values)/len(values) if values else None,
            mean_steps=sum(v['steps'] for v in values)/len(values) if values else None)
        if arm!='NATIVE':
            wins=[i for i,r in rows.items() if r['outcomes'][arm]['success']>r['outcomes']['NATIVE']['success']]
            losses=[i for i,r in rows.items() if r['outcomes'][arm]['success']<r['outcomes']['NATIVE']['success']]
            paired[arm]=dict(wins=wins,losses=losses,win_episode_ids=[manifest[i]['episode_id'] for i in wins],
                loss_episode_ids=[manifest[i]['episode_id'] for i in losses],delta_sr=(len(wins)-len(losses))/n if len(rows)==n else None)
    for h in sorted({r['house'] for r in manifest.values()}):
        selected=[r for r in rows.values() if r['house']==h]
        houses[h]={a:dict(complete=len(selected),planned=sum(e['house']==h for e in manifest.values()),successes=sum(r['outcomes'][a]['success'] for r in selected)) for a in arms}
    return dict(complete_groups=len(rows),planned_groups=n,arms=arms,paired=paired,by_house=houses,
        split=protocol['split'],evaluation_complete=len(rows)==n,full_1839=False,adopted=False)


def main(root):
    lock=(root/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(root/'train');p=u.read(root/'PROTOCOL.json');workers=[];leased=False;started=time.time()
    receipts=[u.read(x) for x in (root/'attempts').glob('*.json')]
    hours=sum(r['wall_seconds']/3600 for r in receipts if r['gpu'] is not None)
    wall_before=sum(r.get('wall_seconds',0) for r in [u.read(x) for x in (root/'sessions').glob('*.json')])
    state=dict(status='RUNNING',phase='COLLECT_TRAIN',started_unix=started,gpu_hours=hours);phase_run=root/'train'

    def save():
        planned=u.read(phase_run/'PROTOCOL.json')['planned_groups']
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in workers if w['gpu'] is not None),
            planned_groups=planned,recorded_groups=len(records(phase_run,'evaluation',False)),sealed_groups=len(records(phase_run,'evaluation',True)),
            workers=[dict(pid=w['proc'].pid,gpu=w['gpu'],output=str(w['output']),
                progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(root/'STATUS.json',state)

    def finish(w):
        nonlocal hours
        elapsed=time.time()-w['start'];hours+=elapsed/3600 if w['gpu'] is not None else 0;w['log'].close()
        u.write(w['receipt'],dict(gpu=w['gpu'],uuid=w['uuid'],pid=w['proc'].pid,command=w['command'],output=str(w['output']),
            returncode=w['proc'].returncode,wall_seconds=elapsed))

    def spawn(command,gpu,output):
        path=root/'attempts'/f'{state["phase"]}_{gpu}_{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
        log=path.open('x');uuid=devices()[gpu]['uuid'] if gpu is not None else ''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        proc=subprocess.Popen(command,cwd=u.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(proc=proc,gpu=gpu,uuid=uuid,output=output,command=command,log=log,receipt=path.with_suffix('.json'),start=time.time()))

    def wait():
        last_space=0
        while workers:
            for w in list(workers):
                if w['proc'].poll() is not None:
                    finish(w);workers.remove(w)
                    if w['proc'].returncode:raise RuntimeError('WORKER_FAILED:'+str(w['receipt']))
            save()
            if state['gpu_hours']>p['gpu_session_hours_limit'] or wall_before+time.time()-started>p['wall_hours_limit']*3600:
                raise RuntimeError('REGISTERED_BUDGET_EXHAUSTED')
            if time.time()-last_space>60:
                total=sum(f.stat().st_size for f in root.rglob('*') if f.is_file());state['artifact_bytes']=total;last_space=time.time()
                if total>p['artifact_limit_gib']*1024**3:raise RuntimeError('ARTIFACT_BUDGET_EXHAUSTED')
            if workers:time.sleep(5)

    def collect(run):
        nonlocal phase_run
        phase_run=run;u.verify_sources(run)
        for session in (run/'evaluation').glob('*'):
            if not (session/'STATE_SEAL.json').exists():
                target=run/'failed_attempts';target.mkdir(exist_ok=True);session.rename(target/(session.name+'_'+str(time.time_ns())))
        planned=[e['id'] for e in u.read(run/'DATA_MANIFEST.json')['episodes']]
        while True:
            done=records(run,'evaluation',True);pending=[i for i in planned if i not in done]
            if not pending:break
            inventory=devices();free=[g for g in p['gpu_indices'] if inventory[g]['free_mib']>=26000]
            if not free:raise RuntimeError('NO_AUTHORIZED_GPU_WITH_MEMORY')
            chunk=p['first_wave_pairs_per_gpu'] if not done else p['chunk_pairs']
            for rank,gpu in enumerate(free):
                shard=pending[rank::len(free)][:chunk]
                if not shard:continue
                out=run/'evaluation'/f'gpu{gpu}_{time.time_ns()}'
                spawn([str(u.PYTHON),'-I','-B',str(HERE/'worker.py'),'--run',str(run),'--output',str(out),'--ids',','.join(map(str,shard))],gpu,out)
            wait();u.write(run/'LIVE_RESULT.json',summarize(run))
        result=summarize(run);assert result['evaluation_complete'];u.write(run/'RESULT.json',result)

    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        before=placeholder('status')
        if before['leases'] or before['manual_paused'] or before['external_pids']:raise RuntimeError('GPU_RESOURCE_BUSY')
        acquired=placeholder('acquire');leased=True
        u.write(root/f'RESOURCE_LEASE_{time.time_ns()}.json',dict(before=before,acquired=acquired,pid=os.getpid()))
        collect(root/'train')
        state['phase']='FIT_GATES';save()
        if not (root/'GATE_REVIEW.json').exists():
            spawn([str(u.PYTHON),'-I','-B',str(HERE/'train.py'),'--run',str(root)],None,root);wait()
        gate_review=u.read(root/'GATE_REVIEW.json');state['actual_gate_steps']=gate_review['actual_total_steps']
        unseen=prepare_unseen(root)
        if unseen is None:
            state.update(status='DATA_LIMITED',phase='NO_TRAINABLE_GAIN_HARM_GATE')
            u.write(root/'RESULT.json',dict(status='DATA_LIMITED',gate_review=gate_review,unseen_executions=0,new_unseen_sr=None))
            (root/'REPORT_ZH.md').write_text('DATA_LIMITED\n\n新 TRAIN 采集已完成，但真实类别覆盖仍不足；没有训练可部署选择器，没有新增 unseen 结果。全部样本与缺项保留。\n')
        else:
            state['phase']='LIVE_UNSEEN';collect(unseen)
            result=summarize(unseen)
            result.update(status='COMPLETE',gpu_hours=hours,gate_review=gate_review,unseen_previously_exposed=True,
                expected_gate_arms=['BC_GATE','B2_GATE','OURS_GATE'],
                gates_not_run=[a+'_GATE' for a,r in gate_review['results'].items() if r['status']!='TRAINED'],
                limitation='Frozen old heads used EU6. Fixed200 excludes EU6; neither complete1839 nor a blind test.',
                architectural_prototype_evaluated=False,method_benefit='SEE_PAIRED_RESULTS_NOT_AUTOMATIC_ADOPTION')
            u.write(root/'RESULT.json',result)
            lines=['COMPLETE','','固定已暴露官方 unseen 子集 200 条；本轮所有已注册策略真实运行，同组前缀审计与参数封存通过。',
                '旧头训练含 EU6，本清单排除 EU6；不是完整 1839 条、不是盲测。选择器仅用 TRAIN；没有按 DEV/unseen 选择阈值。','']
            for a,r in result['arms'].items():lines.append(f'{a}: {r["successes"]}/{r["planned"]}, SR={r["sr"]:.4f}, SPL={r["spl"]:.4f}')
            lines+=['',str(result['paired']),'','缺失 gate：'+str(result['gates_not_run']), '新 evidence_arch_v1 架构未在本轮训练/验证。']
            (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');state.update(status='COMPLETE',phase='UNSEEN_REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(root/f'FAILURE_{time.time_ns()}.json',dict(state,traceback=traceback.format_exc()));raise
    finally:
        for w in workers:
            if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:os.killpg(w['proc'].pid,signal.SIGTERM)
        for w in workers:
            try:w['proc'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:os.killpg(w['proc'].pid,signal.SIGKILL)
                w['proc'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:u.write(root/f'PLACEHOLDER_RELEASE_{time.time_ns()}.json',dict(result=placeholder('release'),unix=time.time()))
        session=root/'sessions';session.mkdir(exist_ok=True)
        u.write(session/f'{time.time_ns()}.json',dict(wall_seconds=time.time()-started,status=state['status']))
        state['finished_unix']=time.time();save();lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    root=HERE/'runs'/a.run_id
    if (root/'STARTED.json').exists() and not a.resume:raise ValueError('RESUME_REQUIRED')
    if u.read(root/'STATUS.json')['status'] in ('COMPLETE','DATA_LIMITED'):raise ValueError('CLOSED_RUN_READ_ONLY')
    u.write(root/'STARTED.json',dict(unix=time.time(),resume=a.resume));main(root)
