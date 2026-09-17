"""Independent recovery evidence acceptance; never mutates an old restoration result."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARALLEL=HERE.parent
LINE=PARALLEL.parents[1]
ROOT=LINE.parents[1]
RECOVERY=PARALLEL/'holder_recovery_v2'
EXPECTED_RECOVERY_LOCK='8e0a0e222ec3ef10dcd8cd447def24ce3e23fcfd3c8156204ab735472c25085d'


def sha(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return json.loads(path.read_text())


def validate_evidence(gpu,result,before,active,pre,recovery):
    assert gpu in (6,7),'THIS_MERGE_ONLY_ORIGINAL_PARALLEL_GPU67'
    assert result['restoration'].get('restored') is False
    assert result['restoration'].get('error')=="AssertionError('SLEEPER_IDENTITY_CHANGED')"
    assert result['error'] is None
    assert recovery['gpu']==pre['gpu']==gpu
    assert recovery['old_receipts_unchanged'] is True and recovery['old_runtime_result_overridden'] is False
    assert recovery['external_processes_stopped']==0
    assert pre['holder_identity']==before['identity']
    assert pre['frozen_empty_argv']==active['sleeper']
    assert active['sleeper']['cmdline']==['']
    corrected=pre['independently_verified_sleeper']
    assert corrected==dict(active['sleeper'],cmdline=['sleep','24000'])
    restored=recovery['restoration'];assert restored['restored'] is True
    old=before['identity'];new=restored['new_identity']
    assert type(new['pid']) is int and new['pid']>0 and new['pid']!=corrected['pid']
    assert new['starttime_ticks']>corrected['starttime_ticks']
    for key in ('proc_uid','cwd','cmdline'):assert new[key]==old[key],('RESTORED_IDENTITY_MISMATCH',key)
    snapshot=restored['gpu']
    assert snapshot['gpu']==gpu and snapshot['uuid']==old['gpu_uuid']
    assert snapshot['processes'].get(str(new['pid']),snapshot['processes'].get(new['pid'],0))>20000
    assert not restored.get('recovery_pane_preserved_if_failed',False)
    return dict(kind='EXTERNAL_VERSIONED_RESTORATION_EVIDENCE',gpu=gpu,
        original_restored=False,recovery_restored=True,new_identity=new,
        original_failure_preserved=True)


def verify(attempt,result,gpu):
    if result['restoration'].get('restored'):
        return dict(kind='ORIGINAL_RUNTIME_RESTORATION',gpu=gpu,original_restored=True)
    assert attempt.resolve()==PARALLEL/'runtime_v1/lanes'/f'gpu_{gpu}/attempt_000'
    assert sha(RECOVERY/'INPUT_LOCK.json')==EXPECTED_RECOVERY_LOCK
    for path,h in read(RECOVERY/'INPUT_LOCK.json').items():assert sha(ROOT/path)==h,path
    recovered=RECOVERY/'runs'/f'gpu_{gpu}_attempt_000'
    pre=read(recovered/'PRE_RECOVERY.json');recovery=read(recovered/'RESULT.json')
    assert pre['original_attempt']==str(attempt.relative_to(ROOT))
    expected_files={str((attempt/name).relative_to(ROOT)) for name in
        ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json')}
    assert set(pre['original_receipt_sha256'])==expected_files
    for path,h in pre['original_receipt_sha256'].items():assert sha(ROOT/path)==h,('OLD_RECEIPT_CHANGED',path)
    assert read(attempt/'RESULT.json')==result
    assert read(attempt/'RESTORATION.json')==result['restoration']
    approval=RECOVERY/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json'
    assert sha(approval)==pre['approval_sha256']
    assert read(approval)==dict(approved=True,node='EMPTY_ARGV_SAME_SLEEPER_NATURAL_CLOSE_RECOVERY_V2',
        gpu=gpu,input_lock_sha256=EXPECTED_RECOVERY_LOCK)
    evidence=validate_evidence(gpu,result,read(attempt/'LEASE_BEFORE.json'),read(attempt/'LEASE_ACTIVE.json'),pre,recovery)
    files=[recovered/'PRE_RECOVERY.json',recovered/'RESULT.json',approval,RECOVERY/'INPUT_LOCK.json']
    evidence['verified_hashes']={str(p.relative_to(ROOT)):sha(p) for p in files}
    evidence['original_receipt_sha256']=pre['original_receipt_sha256']
    return evidence
