"""Exact main-verified GPU7 sleeper recovery after zero-worker startup failure."""
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
RUNTIME=ROOT/'projects/qwen35_indoor_nav/data_pipeline/ordinary_fullscale_source_v1/runtime_v3'
sys.path.insert(0,str(RUNTIME))
import common as c
import run as ops
FROZEN_SLEEP=dict(pid=3265530,starttime_ticks=132082157,proc_uid=0,cwd=str(ROOT),cmdline=['sleep','24000'])

def execute():
    assert c.ROOT==ROOT
    c.approved(7)
    lane=RUNTIME/'lanes/gpu_7'
    producer=(lane/'PRODUCER.lock').open('a');fcntl.flock(producer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=lane/'attempt_000'
    result=c.read(attempt/'RESULT.json')
    assert result['error']=="AssertionError('UNEXPECTED_NEW_SLEEPER_COMMAND')" and result['workers']==[]
    assert result['restoration']==c.read(attempt/'RESTORATION.json')
    assert result['restoration']==dict(restored=False,error="AssertionError('PANE_DEAD_TRANSITION_TIMEOUT')",external_processes_stopped=0)
    assert not list(attempt.glob('PROCESS_*.json')) and not (attempt/'LEASE_ACTIVE.json').exists()
    identity=next(r for r in c.read(c.IDENTITIES)['holders'] if r['gpu_device']==7)
    before=c.read(attempt/'LEASE_BEFORE.json');assert before['identity']==identity
    for shard in c.LANES[7]:
        state=c.state(shard);assert state['terminal_jobs']==0 and not (c.shard_root(shard)/'routes').exists()
    for _ in range(2):
        assert ops.process_identity(FROZEN_SLEEP['pid'])==FROZEN_SLEEP,'EXACT_VERIFIED_SLEEPER_CHANGED'
        assert ops.pane(identity,'#{pane_pid}')==str(FROZEN_SLEEP['pid'])
        assert ops.pane(identity,'#{pane_id}')==identity['pane_id']
        assert ops.pane(identity,'#{pane_dead}')=='0'
        assert ops.pane(identity,'#{pane_start_command}')=='sleep 24000'
        assert ops.pane(identity,'#{pane_current_path}')==str(ROOT)
        time.sleep(.1)
    gpu=ops.gpu_snapshot(identity);ops.contexts(gpu,restorable=True)
    files=[attempt/name for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','EXECUTION_CONFIG.json')]
    hashes={str(p.relative_to(ROOT)):c.sha(p) for p in files}
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    c.save(out/'PRE_RECOVERY.json',dict(gpu=7,frozen_sleeper=FROZEN_SLEEP,original_holder=identity,
        raw_failure=result,original_hashes=hashes,zero_worker_attempt=True,gpu_before=gpu,
        authority='user permits restoring verified idle holders; main independently verified exact current sleeper',
        script_sha256=c.sha(Path(__file__).resolve())))
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    restored=None
    try:
        for s in handlers:signal.signal(s,signal.SIG_IGN)
        try:restored=ops.restore_holder(identity,FROZEN_SLEEP,before['remain'],out,released_pid=None)
        except BaseException as exc:restored=dict(restored=False,error=repr(exc),external_processes_stopped=0)
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
    unchanged=all(c.sha(ROOT/p)==h for p,h in hashes.items())
    final=dict(gpu=7,restoration=restored,old_receipts_unchanged=unchanged,zero_worker_attempt=True,
               external_processes_stopped=0,scientific_pass=False)
    c.save(out/'RESULT.json',final);print(json.dumps(final),flush=True)
    assert restored['restored'] and unchanged

if __name__=='__main__':execute()
