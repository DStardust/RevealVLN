"""Main-agent-only exact interrupted-route quarantine; no GPU or retry."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
JOB = 'r2r_383f0490ffb5248c87ed'
EXPECTED = {'replay_certificate.json': 'c33ca6282a578811e5765b83667422210be0d91e012807af4af24421080f3ab4'}

def terminal():
    return dict(job_id=JOB, scene_id='29hnd4uzFmX', status='RESOURCE_INTERRUPTED',
                reason='V2_EXTERNAL_CONTEXT_WATCHDOG_INTERRUPTION', instruction_records=0,
                training_eligible=False, retry_allowed=False,
                quarantine_directory='recovery_v3/interrupted/'+JOB)

def main():
    guard = (BASE/'PRODUCER.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('v3_preflight', HERE/'preflight.py')
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    preflight.verify(prepared=False)
    source = BASE/'routes'/JOB
    assert source.resolve().is_relative_to(ROOT) and not source.is_symlink()
    files = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in source.rglob('*') if p.is_file()}
    assert files == EXPECTED
    assert all(not p.is_symlink() for p in source.rglob('*'))
    target = HERE/'interrupted'/JOB
    target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    manifest = dict(job_id=JOB, original_path=str(source.relative_to(ROOT)),
                    quarantined_path=str(target.relative_to(ROOT)), files_sha256=files,
                    reason='RESOURCE_INTERRUPTED_NOT_A_SCIENTIFIC_FAILURE', original_files_preserved=True)
    with (HERE/'INTERRUPTION_MANIFEST.json').open('x') as handle:
        json.dump(manifest,handle,indent=2)
        handle.flush(); os.fsync(handle.fileno())
    source.rename(target)
    source.mkdir(exist_ok=False)
    with (source/'result.json').open('x') as handle:
        json.dump(terminal(),handle,indent=2)
        handle.flush(); os.fsync(handle.fileno())
    with (BASE/'LEDGER.jsonl').open('a') as handle:
        handle.write(json.dumps(terminal())+'\n')
        handle.flush(); os.fsync(handle.fileno())
    preflight.verify(prepared=True)
    print(json.dumps({'quarantine_prepared':True,'terminal_routes':68,'retry_allowed':False,'gpu_operations':0}))

if __name__ == '__main__':
    main()
