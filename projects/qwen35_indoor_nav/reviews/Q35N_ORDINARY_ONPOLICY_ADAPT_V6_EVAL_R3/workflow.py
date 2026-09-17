"""One evaluation only after registered zero-action transport failure; no GPU3/4/5 control."""
import fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_onpolicy_adapt_dev_v6r3'
PARENT=HERE.parent/'Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1/workflow.py'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(name,**extra):
    p=HERE/'WORKFLOW_STATUS.tmp';p.write_text(json.dumps(dict(status=name,unix=time.time(),automatic_next_training=False,**extra),ensure_ascii=False));p.replace(HERE/'WORKFLOW_STATUS.json')
def freeze():
    assert sha(PARENT)=='7f468b13cb855fd63a99e2a0c6a9de4be892ece2e99dd20dbf96be7f68ade18a'
    paths=[Path(x) for x in read(CASE/'SOURCE_LOCK.json')['files']]
    paths += list(HERE.glob('*.py'))+[CASE/'SOURCE_LOCK.json',CASE/'MAIN_REVIEW.json',PARENT]
    save(HERE/'SEAL.json',dict(files={str(p):sha(p) for p in paths},training_allowed=False,
        gate=dict(sr_delta_gt=0,spl_delta_ge=0,ndtw_delta_ge=-.01),one_attempt=True))
    print('FROZEN_EVALUATION_ONLY_WORKFLOW')
def main():
    lock=(HERE/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'WORKFLOW_RESULT.json').exists()
    for p,d in read(HERE/'SEAL.json')['files'].items():assert sha(p)==d,p
    def interrupted(sig,frame):raise RuntimeError('WORKFLOW_SIGNAL_'+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    child=None
    try:
        status('EVALUATING_FIXED_CORRECTION_1000')
        with (HERE/'eval_launcher.log').open('x') as log:
            child=subprocess.Popen([str(PY),'-I','-S','-B',str(CASE/'launch.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            began=time.monotonic()
            while child.poll() is None:
                assert time.monotonic()-began<5520,'EVAL_WALL_LIMIT'
                status('EVALUATING_FIXED_CORRECTION_1000',own_launcher_pid=child.pid,elapsed_seconds=time.monotonic()-began);time.sleep(10)
            assert child.returncode==0,'EVALUATION_OR_TRACE_AUDIT_FAILED'
        status('ANALYZING_PAIRED_RESULTS')
        assert sha(PARENT)=='7f468b13cb855fd63a99e2a0c6a9de4be892ece2e99dd20dbf96be7f68ade18a'
        s=importlib.util.spec_from_file_location('unchanged_r2r_report',PARENT);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        m.HERE=HERE;m.CASE=CASE
        result=m.close_report()
        value=dict(status='COMPLETE',unix=time.time(),positive_development_signal=result['positive_development_signal'],
            paired_delta=result['paired']['delta'],scientific_gain_verified=False,automatic_next_training=False,
            predecessor_zero_action_failure_preserved=True)
        save(HERE/'WORKFLOW_RESULT.json',value);status('COMPLETE',positive_development_signal=value['positive_development_signal'])
    except BaseException as exc:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=60)
        value=dict(status='FAILED_OR_BLOCKED',unix=time.time(),error=repr(exc),traceback=traceback.format_exc(),
            training_processes_signaled=[],foreign_processes_signaled=[],automatic_retry=False)
        save(HERE/'WORKFLOW_RESULT.json',value);status('FAILED_OR_BLOCKED',error=repr(exc));raise
    finally:lock.close()
if __name__=='__main__':
    assert sys.argv[1:] in (['freeze'],['run'])
    freeze() if sys.argv[1]=='freeze' else main()
