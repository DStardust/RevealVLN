"""Wait for the known metadata-scan failure; never interrupt a running evaluator."""
import json,os,subprocess,sys,time
from pathlib import Path
OPS=Path(__file__).resolve().parent;BASE=OPS.parent;RUN=BASE/'runs/contrast_001'
def eligible(status,service_state,result_exists):
    return not result_exists and status.get('status')=='STOPPED_WITH_EVIDENCE' and status.get('reason')=="TimeoutError('DISK_SCAN_DEADLINE')" and service_state in ('failed','inactive')
def write(value):
    temp=OPS/'GUARD_STATUS.tmp';temp.write_text(json.dumps(dict(unix=time.time(),**value),indent=2)+'\n');temp.replace(OPS/'GUARD_STATUS.json')
def main():
    deadline=time.monotonic()+24*3600
    unit='q35n-ordinary-contrast20-20260924-02.service'
    while time.monotonic()<deadline:
        status=json.loads((RUN/'STATUS.json').read_text());result=(RUN/'RESULT.json').exists()
        state=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True).strip()
        if result:write(dict(status='DONE_WITHOUT_INTERVENTION'));return
        if eligible(status,state,result):
            name='ordinary-contrast20-20260924-resource-r1'
            cmd=[sys.executable,'-I','-S','-B',str(BASE/'standalone.py'),'start',name,'--',sys.executable,'-I','-S','-B',str(OPS/'pipeline.py'),'--run-id','contrast_001','--resume']
            # Job directories refuse duplicates. Only the same recorded resource failure is retryable.
            env=dict(os.environ,V16_STANDALONE_PYTHON=sys.executable)
            process=subprocess.run(cmd,capture_output=True,text=True,env=env)
            write(dict(status='RESUME_SUBMITTED' if process.returncode==0 else 'RESUME_SUBMISSION_FAILED',command=cmd,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
            if process.returncode:raise RuntimeError('RECOVERY_SUBMISSION_FAILED')
            (BASE/'LAST_JOB.txt').write_text(name+'\n');return
        if status.get('status')=='STOPPED_WITH_EVIDENCE' and state in ('failed','inactive'):
            write(dict(status='OTHER_FAILURE_REQUIRES_REVIEW',failure=status));return
        write(dict(status='WAITING_NO_CHANGES_TO_ACTIVE_RUN',source_unit=unit,phase=status.get('phase')));time.sleep(30)
    write(dict(status='GUARD_WINDOW_COMPLETE_NO_INTERVENTION'))
if __name__=='__main__':main()
