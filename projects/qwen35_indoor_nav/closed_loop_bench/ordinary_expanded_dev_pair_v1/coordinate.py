"""Fixed before/after inference, no automatic training continuation or retry."""
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('dev_preparation',HERE/'prepare_cases.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
PY=p.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def status(state,**kw):
    temp=HERE/'STATUS.pending'
    with temp.open('w') as f:json.dump(dict(status=state,unix=time.time(),automatic_long_training=False,**kw),f,allow_nan=False)
    temp.replace(HERE/'STATUS.json')


def run_case(role,checkpoint):
    case=p.bind_case(role,checkpoint)
    with (case/'controller.log').open('x') as log:
        child=subprocess.Popen([str(PY),'-I','-S','-B','-u',str(case/'launch.py')],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+4200
            while child.poll() is None:
                assert time.monotonic()<deadline,'CASE_CONTROLLER_WALL'
                progress_path=case/'run_001/PROGRESS.json'
                current=p.read(progress_path) if progress_path.exists() else {}
                status('EVALUATING_'+role.upper(),role=role,launcher_pid=child.pid,progress=current)
                time.sleep(5)
            assert child.returncode==0,'DEV_CASE_FAILED:'+role
            result=p.read(case/'run_001/RESULT.json');assert result['status']=='COMPLETE' and result['completed']==100
            return result
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=60)


def pair_report(before,after):
    def rows(role):
        path=HERE.parent/f'ordinary_expanded_dev_{role}_v1/run_001/episodes.csv'
        return list(csv.DictReader(path.open()))
    a=rows('before');b=rows('after');assert len(a)==len(b)==100
    changed=[]
    for old,new in zip(a,b):
        assert all(old[k]==new[k] for k in ('index','episode_id','trajectory_id','house'))
        changed.append(dict(episode_id=old['episode_id'],house=old['house'],
                            delta_sr=float(new['success'])-float(old['success']),delta_spl=float(new['spl'])-float(old['spl'])))
    sr=after['sr']-before['sr'];spl=after['spl']-before['spl']
    result=dict(status='PAIRED_DEV_COMPLETE',before=before,after=after,delta_sr=sr,delta_spl=spl,
                improved_routes=sum(x['delta_sr']>0 for x in changed),worsened_routes=sum(x['delta_sr']<0 for x in changed),
                paired_episodes=changed,by_house={h:dict(delta_sr=after['by_house'][h]['sr']-before['by_house'][h]['sr'],
                   delta_spl=after['by_house'][h]['spl']-before['by_house'][h]['spl']) for h in before['by_house']},
                development_improvement_signal=sr>0 and spl>=-.02,scientific_gain_verified=False,
                automatic_long_training=False,decision='MAIN_REVIEW_BEFORE_ANY_NEXT_TRAINING',unix=time.time())
    p.save(HERE/'RESULT.json',result)
    return result


def main():
    lock=(HERE/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'STATUS.json').exists(),'ONE_ATTEMPT_ONLY'
    for file,digest in p.read(HERE/'CODE_FREEZE.json').items():assert p.sha(Path(file))==digest,'PAIR_SOURCE_CHANGED'
    tests=p.read(HERE/'CPU_TEST_RESULT.json');assert tests['passed']
    status('STARTING_BEFORE')
    def interrupt(signum,frame):raise RuntimeError('PAIR_SIGNAL_'+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupt)
    signal.alarm(12600)
    try:
        before=run_case('before',p.LINE/'sft_acceptance/ordinary_sync_recovery_v1/formal/attempt_001/checkpoint_000051301.pt')
        deadline=time.monotonic()+4800
        while not (p.TRAIN/'lease_v1/LEASE_RESULT.json').exists():
            assert time.monotonic()<deadline,'TRAINING_WAIT_BUDGET'
            status('WAITING_FOR_BOUNDED_PILOT_CLOSURE',before_sr=before['sr'],before_spl=before['spl'])
            time.sleep(5)
        result=p.read(p.TRAIN/'formal/attempt_001/RESULT.json')
        after=run_case('after',Path(result['latest_checkpoint']))
        summary=pair_report(before,after)
        status('PAIRED_DEV_COMPLETE',delta_sr=summary['delta_sr'],delta_spl=summary['delta_spl'],
               development_improvement_signal=summary['development_improvement_signal'])
    except BaseException as exc:
        failure=dict(status='FAILED_OR_BLOCKED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc(),scientific_gain_verified=False)
        p.save(HERE/'FAILURE.json',failure);status(failure['status'],error=repr(exc));raise
    finally:signal.alarm(0);lock.close()


if __name__=='__main__':main()
