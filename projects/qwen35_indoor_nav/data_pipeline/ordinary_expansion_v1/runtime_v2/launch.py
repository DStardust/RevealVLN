"""One-shot three independent leased lanes, then bounded CPU merge; no training."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c

def main():
    c.immutable_verify()
    for gpu in c.LANES:
        assert c.read(HERE/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')==c.approval_value(gpu)
    lock=(HERE/'LAUNCHER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=HERE/'launch_001';out.mkdir(exist_ok=False)
    started=time.monotonic();children={};logs=[];error=None;merge=None
    def interrupt(sig,frame):raise InterruptedError('USER_STOP:'+str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for gpu in c.LANES:
            log=(out/f'gpu_{gpu}.log').open('x');logs.append(log)
            argv=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'run.py'),'--gpu',str(gpu)]
            child=subprocess.Popen(argv,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[gpu]=child
            c.save(out/f'GPU_{gpu}_PROCESS.json',dict(pid=child.pid,argv=argv,started_unix=time.time()))
        while any(p.poll() is None for p in children.values()):
            assert time.monotonic()-started<43500,'TOTAL_LANE_DEADLINE'
            time.sleep(3)
        results={g:c.read(HERE/f'lanes/gpu_{g}/attempt_000/RESULT.json') for g in c.LANES}
        if all(p.returncode==0 for p in children.values()) and all(r['error'] is None and r['restoration'].get('restored') for r in results.values()):
            log=(out/'merge.log').open('x');logs.append(log)
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            merge=subprocess.Popen([str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'merge.py')],cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            merge.wait(timeout=7200)
            assert merge.returncode==0,'FINAL_CPU_MERGE_FAILED'
        else:
            raise AssertionError('LANE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY')
    except BaseException as exc:
        error=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        for child in children.values():
            if child.poll() is None:
                child.terminate()  # owned lane supervisor: it cleans only its worker and restores its holder
        for child in children.values():
            if child.poll() is None:
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:error=(error or '')+'; OWN_LANE_CLEANUP_PENDING_NO_FORCED_LEASE_KILL'
        if merge and merge.poll() is None:
            merge.terminate()
            try:merge.wait(timeout=30)
            except subprocess.TimeoutExpired:merge.kill();merge.wait(timeout=10)
        for log in logs:log.close()
        c.save(out/'RESULT.json',dict(error=error,lanes={g:p.poll() for g,p in children.items()},
            merge_returncode=merge.poll() if merge else None,wall_seconds=time.monotonic()-started,
            automatic_retry=False,training_started=False,external_processes_stopped=0))
    if error:raise SystemExit(1)

if __name__=='__main__':main()
