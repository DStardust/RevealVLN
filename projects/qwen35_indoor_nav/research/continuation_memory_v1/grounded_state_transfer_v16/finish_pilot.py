"""Run the FIT-only numerical pilot after this task's collector releases its GPU."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from pipeline import execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--wait-job',required=True)
    args=p.parse_args();run=args.run
    from standalone import folder
    job=folder(args.wait_job);spec=read(job/'JOB.json')
    if spec['uid']!=os.getuid():raise ValueError('FOREIGN_JOB')
    deadline=time.monotonic()+3600
    while True:
        state=read(job/'STATUS.json') if (job/'STATUS.json').exists() else dict(status='PENDING')
        if state['status'] in ('COMPLETE','FAILED','INTERRUPTED'):break
        if time.monotonic()>deadline:raise RuntimeError('OWN_COLLECTOR_WAIT_LIMIT')
        time.sleep(5)
    config=read(run/'PROTOCOL.json');verify_lock(read(run/'SOURCE_LOCK.json'))
    execute(run,config,'raw_audit',[HERE/'audit_data.py',run],LINE/'.envs/q35n_qwen_g2_v1/bin/python3',gpu=False)
    execute(run,config,'golden_pilot',[HERE/'numerical_pilot.py',run],LINE/'.envs/q35n_qwen_g2_v1/bin/python3')
    write(run/'PILOT_COMPLETE.json',dict(numerical='numerical_pilot/RESULT.json',raw_audit='RAW_DATA_AUDIT.json',method_training_started=False),True)

if __name__=='__main__':main()
