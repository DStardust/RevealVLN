import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
JOB = 'r2r_383f0490ffb5248c87ed'

def verify(prepared=True):
    lock = json.loads((HERE/'INPUT_LOCK.json').read_text())
    for name, digest in lock['immutable'].items():
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    ledger = (BASE/'LEDGER.jsonl').read_bytes()
    prefix = lock['ledger_prefix']
    assert hashlib.sha256(ledger[:prefix['bytes']]).hexdigest() == prefix['sha256']
    rows = [json.loads(row) for row in ledger.splitlines()]
    ids = {row['job_id'] for row in rows}
    assert len(ids) == len(rows) == (68 if prepared else 67)
    if prepared:
        expected = dict(job_id=JOB,scene_id='29hnd4uzFmX',status='RESOURCE_INTERRUPTED',
            reason='V2_EXTERNAL_CONTEXT_WATCHDOG_INTERRUPTION',instruction_records=0,
            training_eligible=False,retry_allowed=False,
            quarantine_directory='recovery_v3/interrupted/'+JOB)
        assert rows[-1] == expected
        route = BASE/'routes'/JOB
        assert {p.name for p in route.iterdir()} == {'result.json'}
        assert json.loads((route/'result.json').read_text()) == expected
        original = HERE/'interrupted'/JOB
        assert {p.name for p in original.iterdir()} == {'replay_certificate.json'}
        assert hashlib.sha256((original/'replay_certificate.json').read_bytes()).hexdigest() == 'c33ca6282a578811e5765b83667422210be0d91e012807af4af24421080f3ab4'
    else:
        assert len(ledger) == prefix['bytes']
    jobs = json.loads((BASE/'JOBS.json').read_text())
    assert len(jobs) == 1000
    assert ids <= {job['job_id'] for job in jobs}
    unfinished = {p.name for p in (BASE/'routes').iterdir() if p.is_dir() and p.name not in ids}
    assert unfinished == (set() if prepared else {JOB}), ('UNFINISHED_ROUTE_NO_OVERWRITE', unfinished)
    return lock

if __name__ == '__main__':
    print(json.dumps({'preflight_pass':bool(verify()),'gpu_operations':0}))
