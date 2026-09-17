"""One recovery lane then CPU merge; preserve any failure, never retry or train."""
import json
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
    c.approved(3);out=HERE/'launch_001';out.mkdir(exist_ok=False)
    started=time.monotonic();proc=None;phase=None;error=None;results=[]
    def interrupt(sig,frame):raise InterruptedError(str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        stages=[('production',[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'run.py'),'--gpu','3'],43500),
                ('strict_merge',[str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'merge.py')],7200)]
        for phase,argv,limit in stages:
            with (out/(phase+'.log')).open('x') as log:
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                proc=subprocess.Popen(argv,cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                c.save(out/(phase+'_PROCESS.json'),dict(pid=proc.pid,argv=argv,unix=time.time()))
                proc.wait(timeout=limit)
                results.append(dict(phase=phase,returncode=proc.returncode))
                assert proc.returncode==0,phase+'_FAILED_NO_RETRY'
    except BaseException as exc:error=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=180 if phase=='production' else 30)
            except subprocess.TimeoutExpired:
                if phase=='strict_merge':proc.kill();proc.wait(timeout=10)
                else:error=(error or '')+'; LANE_CLEANUP_PENDING_NO_FORCED_LEASE_KILL'
        c.save(out/'RESULT.json',dict(error=error,phases=results,wall_seconds=time.monotonic()-started,automatic_retry=False,training_started=False))
    if error:raise SystemExit(1)

if __name__=='__main__':main()
