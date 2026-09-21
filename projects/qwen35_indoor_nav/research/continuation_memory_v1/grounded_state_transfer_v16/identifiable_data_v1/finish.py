"""Independent bounded postprocessing for the registered collection service."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--job',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    run=HERE/'runs'/a.run_id;job=HERE/'standalone_jobs'/a.job;out=HERE/a.output
    if not run.is_dir() or not job.is_dir() or out.exists():raise ValueError('REGISTERED_JOB_OR_OUTPUT')
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (HERE/'review.py',HERE/'certify.py')}
    began=time.monotonic()
    while read(job/'STATUS.json')['status']=='RUNNING':
        if time.monotonic()-began>11200:raise TimeoutError('POSTPROCESS_WAIT_LIMIT')
        time.sleep(10)
    state=read(run/'STATUS.json')
    if state['status'] not in ('COLLECTION_COMPLETE','COLLECTION_SHORTFALL'):
        raise RuntimeError('COLLECTION_STOPPED_WITHOUT_FINAL_DATASET: '+str(state))
    for path,expected in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('REVIEW_SOURCE_CHANGED')
    subprocess.run([sys.executable,'-I','-B',str(HERE/'review.py'),'--run',str(run),'--output',str(out)],check=True)
    (out/'POSTPROCESS_RECEIPT.json').write_text(json.dumps(dict(job=a.job,collection_state=state,
        review_sources=sources,wait_and_review_seconds=time.monotonic()-began),indent=2)+'\n')

if __name__=='__main__':main()
