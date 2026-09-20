"""Freeze completed physical FIT/DEV assets and autonomously enter the one full protocol."""
import argparse
from pathlib import Path
import sys
import subprocess
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--wait-job',required=True);args=p.parse_args()
    from standalone import folder
    job=folder(args.wait_job);spec=read(job/'JOB.json')
    if spec['uid']!=os.getuid():raise ValueError('FOREIGN_JOB')
    deadline=time.monotonic()+3*3600
    while read(job/'STATUS.json')['status']=='RUNNING':
        if time.monotonic()>deadline:raise RuntimeError('COLLECTOR_WAIT_LIMIT')
        time.sleep(5)
    source=args.source_run;records=[read(p) for p in source.glob('HOUSE_*.json')]
    families=[f for r in records for f in r['families']]
    counts={s:sum(f['split']==s for f in families) for s in ('FIT','DEV')}
    if counts!={'FIT':16,'DEV':2} or not all(r['complete'] for r in records):
        write(source/'PROMOTION_BLOCKED.json',dict(reason='REGISTERED_FIT_DEV_SCALE_INCOMPLETE',actual=counts,expected=dict(FIT=16,DEV=2),method_training_started=False))
        raise RuntimeError('FIT_DEV_PHYSICAL_COVERAGE_INCOMPLETE')
    golden=read(HERE/'runs/v16_fit_golden_003/numerical_pilot/RESULT.json')
    if not golden['base_parameters_unchanged'] or golden['native_argmax_flips'] or not golden['all_processed_inputs_equal']:raise ValueError('GOLDEN_PILOT_NOT_VALID')
    run=HERE/'runs'/args.run_id;run.mkdir(exist_ok=False)
    config=read(HERE/'PROTOCOL.json');immutable(run/'PROTOCOL.json',config)
    for path in source.glob('HOUSE_*.json'):immutable(run/path.name,read(path))
    immutable(run/'REUSED_PHYSICAL_ASSETS.json',dict(source=str(source),prior_protocol_sha256=sha(source/'PROTOCOL.json'),
        source_house_records={p.name:sha(p) for p in source.glob('HOUSE_*.json')},policy_models_used_for_selection=False,
        treatment_comparison='All arms share all selected actual trajectories; generator changes are common infrastructure, not method increments.'))
    lock=source_lock();immutable(run/'SOURCE_LOCK.json',lock)
    for relative,expected in lock['files'].items():
        source_path=LINE/relative;destination=run/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source_path.read_bytes())
        if sha(destination)!=expected:raise ValueError('SOURCE_SNAPSHOT_CHANGED')
    immutable(run/'BUDGET_FREEZE.json',dict(gpu_session_hours=16,max_continuous_hours=3,
        golden_forward_seconds=golden['seconds'],golden_calls=golden['real_qwen_forwards'],
        caveat='Golden pilot includes loading and cold compilation; not a guarantee of full-data throughput. Complete V15 timings retained as planning references. Includes all prior attempts.'))
    command=[str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'pipeline.py'),
             '--config',str(run/'PROTOCOL.json'),'--run-id',args.run_id,'--phase','all','--resume']
    write(source/'PROMOTED_TO.json',dict(run=str(run),command=command),True)
    result=subprocess.run(command,stdin=subprocess.DEVNULL,cwd=ROOT)
    publish=[str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'publish.py'),
             '--message','Publish V16 autonomous pipeline outcome and sealed evidence']
    pushed=subprocess.run(publish,stdin=subprocess.DEVNULL,cwd=ROOT)
    if pushed.returncode:write(run/'PUBLICATION_FAILURE.json',dict(returncode=pushed.returncode,evidence_preserved=True))
    if result.returncode:raise SystemExit(result.returncode)
    if pushed.returncode:raise SystemExit(pushed.returncode)

if __name__=='__main__':main()
