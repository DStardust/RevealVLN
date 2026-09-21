"""Independent bounded CPU diagnosis; three seed workers, no model/simulator service."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)

def main():
    cfg=json.loads((HERE/'PROTOCOL.json').read_text());out=HERE/'run_002';began=time.monotonic();workers=[];logs=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    def launch(name,args):
        log=(out/(name+'.log')).open('x');logs.append(log)
        proc=subprocess.Popen([cfg['torch_python'],'-I','-B',str(HERE/args[0])]+args[1:],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        workers.append(proc);return proc
    def stop(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        subprocess.run([cfg['torch_python'],'-I','-B',str(HERE/'prepare.py')],env=env,stdin=subprocess.DEVNULL,check=True,timeout=300)
        for seed in cfg['seeds']:launch('seed_'+str(seed),['worker.py','--seed',str(seed)])
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('WORKER_FAILED; inspect seed logs')
            if time.monotonic()-began>cfg['wall_seconds']:raise TimeoutError('DIAGNOSTIC_WALL_LIMIT')
            save(out/'STATUS.json',dict(status='RUNNING',phase='probes_and_readback',seconds=time.monotonic()-began,workers=[dict(pid=p.pid,returncode=p.poll()) for p in workers]));time.sleep(5)
        if any(p.returncode for p in workers):raise RuntimeError('WORKER_FAILED')
        proc=launch('review',['review.py']);proc.wait(timeout=120)
        if proc.returncode:raise RuntimeError('REVIEW_FAILED')
        save(out/'STATUS.json',dict(status='COMPLETE',phase='review',seconds=time.monotonic()-began,GPU_hours=0,policy_updates=0,new_navigation_episodes=0))
    except BaseException as exc:
        if out.exists():save(out/'STATUS.json',dict(status='FAILED',reason=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
