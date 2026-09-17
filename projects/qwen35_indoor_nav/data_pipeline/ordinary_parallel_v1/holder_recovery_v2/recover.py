"""Main-approved repair of one proven empty-argv sleeper receipt, after natural closure."""
import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
ROOT=LINE.parents[1]
NODE='EMPTY_ARGV_SAME_SLEEPER_NATURAL_CLOSE_RECOVERY_V2'
EXPECTED_RUNTIME_LOCKS={
    6:('ordinary_parallel_v1/runtime_v1','1aa3343a7d176d8eb53587e02adb8e24c3bfc93b7a422e78b63ad298eba856a2'),
    7:('ordinary_parallel_v1/runtime_v1','1aa3343a7d176d8eb53587e02adb8e24c3bfc93b7a422e78b63ad298eba856a2'),
    3:('ordinary_fullscale_source_v1/runtime_v2','6523fc0812df56141c8e10f8b6abf0c2b9bf900c92eaaec50175288988d7f870'),
    4:('ordinary_fullscale_source_v1/runtime_v2','6523fc0812df56141c8e10f8b6abf0c2b9bf900c92eaaec50175288988d7f870')}


def sha(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    return json.loads(path.read_text())


def save(path,value):
    assert path.resolve().is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def runtime_path(gpu):
    assert gpu in EXPECTED_RUNTIME_LOCKS
    relative,expected=EXPECTED_RUNTIME_LOCKS[gpu]
    runtime=LINE/'data_pipeline'/relative
    assert sha(runtime/'INPUT_LOCK.json')==expected,'REGISTERED_RUNTIME_LOCK_CHANGED'
    return runtime


@contextlib.contextmanager
def registered_runtime(gpu):
    """Resolve each lane's actual reviewed module without cross-lane import-cache contamination."""
    runtime=runtime_path(gpu)
    names=('transport','common','run','safe_size')
    previous={name:sys.modules.get(name) for name in names};old_path=list(sys.path)
    for name in names:sys.modules.pop(name,None)
    sys.path.insert(0,str(runtime))
    try:
        c=importlib.import_module('common');ops=importlib.import_module('run')
        assert c.HERE==runtime and ops.HERE==runtime
        c.approved(gpu)
        if gpu==4:assert callable(ops.holder_command) and callable(ops.holder_environment_matches)
        yield c,ops
    finally:
        sys.path[:]=old_path
        for name,old in previous.items():
            sys.modules.pop(name,None)
            if old is not None:sys.modules[name]=old


def approval_value(gpu):
    assert gpu in EXPECTED_RUNTIME_LOCKS
    return dict(approved=True,node=NODE,gpu=gpu,input_lock_sha256=sha(HERE/'INPUT_LOCK.json'))


def verify_approval(gpu):
    for path,h in read(HERE/'INPUT_LOCK.json').items():assert sha(ROOT/path)==h,path
    assert read(HERE/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')==approval_value(gpu),'MAIN_RECOVERY_APPROVAL_REQUIRED'


def closed_receipts(c,gpu):
    attempt=c.HERE/'lanes'/f'gpu_{gpu}'/'attempt_000'
    result=read(attempt/'RESULT.json');restoration=read(attempt/'RESTORATION.json')
    assert result['error'] is None,'NOT_NATURAL_CLOSURE'
    assert result['restoration']==restoration and restoration.get('restored') is False
    assert restoration.get('error')=="AssertionError('SLEEPER_IDENTITY_CHANGED')",'NOT_THE_REGISTERED_EMPTY_ARGV_FAILURE'
    assert {w['shard'] for w in result['workers']}==set(c.LANES[gpu])
    assert all(w['returncode']==0 for w in result['workers']),'WORKER_NOT_SUCCESSFULLY_CLOSED'
    for shard in c.LANES[gpu]:assert c.state(shard)['complete'],'INCOMPLETE_SHARD'
    before=read(attempt/'LEASE_BEFORE.json');active=read(attempt/'LEASE_ACTIVE.json')
    identity=next(r for r in c.read(c.IDENTITIES)['holders'] if r['gpu_device']==gpu)
    assert before['identity']==identity,'ORIGINAL_HOLDER_IDENTITY_CHANGED'
    sleeper=active['sleeper'];assert sleeper['cmdline']==[''],'ONLY_KNOWN_EMPTY_ARGV_RACE'
    assert sleeper['cwd']==str(ROOT) and sleeper['proc_uid']==identity['proc_uid']
    assert sleeper['pid']!=identity['pid'] and sleeper['starttime_ticks']>0
    files=[attempt/name for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json')]
    return attempt,identity,sleeper,before['remain'],{str(p.relative_to(ROOT)):sha(p) for p in files}


def stable_sleeper(ops,identity,frozen):
    corrected=None
    for _ in range(2):
        actual=ops.process_identity(frozen['pid'])
        for key in ('pid','starttime_ticks','proc_uid','cwd'):
            assert actual[key]==frozen[key],('FROZEN_SLEEPER_IDENTITY_MISMATCH',key)
        assert actual['cmdline']==['sleep','24000'],'NOT_EXACT_OWN_SLEEPER'
        assert ops.pane(identity,'#{pane_id}')==identity['pane_id'],'PANE_ID_CHANGED'
        assert int(ops.pane(identity,'#{pane_pid}'))==frozen['pid'],'PANE_PID_CHANGED'
        assert ops.pane(identity,'#{pane_dead}')=='0','SLEEPER_NOT_LIVE'
        assert ops.pane(identity,'#{pane_current_path}')==str(ROOT),'PANE_CWD_CHANGED'
        if corrected is not None:assert actual==corrected,'SLEEPER_NOT_STABLE'
        corrected=actual
        time.sleep(.1)
    return corrected


def perform_restore(ops,identity,corrected,remain,out):
    try:
        # Original function rechecks/drains current external load and exact sleeper before any signal.
        return ops.restore_holder(identity,corrected,remain,out,released_pid=None)
    except BaseException as error:
        return dict(restored=False,error=repr(error),external_processes_stopped=0,
                    recovery_pane_preserved=True)


def execute(gpu):
    verify_approval(gpu)
    with registered_runtime(gpu) as (c,ops):
        import fcntl
        lane=c.HERE/'lanes'/f'gpu_{gpu}'
        handle=(lane/'PRODUCER.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt,identity,frozen,remain,hashes=closed_receipts(c,gpu)
        corrected=stable_sleeper(ops,identity,frozen)
        snapshot=ops.gpu_snapshot(identity);ops.contexts(snapshot,restorable=True)
        # Active or unclosed production never reaches this point, and no existing receipt is edited.
        out=HERE/'runs'/f'gpu_{gpu}_attempt_000';out.parent.mkdir(exist_ok=True);out.mkdir(exist_ok=False)
        save(out/'PRE_RECOVERY.json',dict(node=NODE,gpu=gpu,original_attempt=str(attempt.relative_to(ROOT)),
            original_receipt_sha256=hashes,frozen_empty_argv=frozen,independently_verified_sleeper=corrected,
            holder_identity=identity,gpu_before=snapshot,approval_sha256=sha(HERE/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')))
        for path,h in hashes.items():assert sha(ROOT/path)==h,'RECEIPT_CHANGED_BEFORE_RESTORE'
        # Protect the bounded original restoration sequence after its complete read-only preflight.
        old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
            restored=perform_restore(ops,identity,corrected,remain,out)
        finally:
            for signum,handler in old_handlers.items():signal.signal(signum,handler)
        unchanged=all(sha(ROOT/path)==h for path,h in hashes.items())
        result=dict(node=NODE,gpu=gpu,restoration=restored,old_receipts_unchanged=unchanged,
            changed_scope='only externally verified identical sleeper argv passed to sealed restore function',
            old_runtime_result_overridden=False,external_processes_stopped=0,scientific_pass=False)
        save(out/'RESULT.json',result)
        print(json.dumps(result),flush=True)
        if not restored.get('restored') or not unchanged:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[3,4,6,7],required=True)
    execute(p.parse_args().gpu)
