"""Exact immutable inputs and append-only ledger prefix checked before production."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]

def verify():
    lock = json.loads((HERE / 'INPUT_LOCK.json').read_text())
    for name, digest in lock['immutable'].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    ledger = (BASE / 'LEDGER.jsonl').read_bytes()
    prefix = lock['ledger_prefix']
    assert len(ledger) == prefix['bytes'], 'CONCURRENT_OR_UNREVIEWED_LEDGER_APPEND'
    assert hashlib.sha256(ledger[:prefix['bytes']]).hexdigest() == prefix['sha256']
    rows = [json.loads(row) for row in ledger.splitlines()]
    ids = {row['job_id'] for row in rows}
    assert len(ids) == len(rows) == 60
    jobs = json.loads((BASE / 'JOBS.json').read_text())
    assert len(jobs) == 1000
    assert ids <= {job['job_id'] for job in jobs}
    unfinished = [p.name for p in (BASE / 'routes').iterdir() if p.is_dir() and p.name not in ids]
    assert not unfinished, ('UNFINISHED_ROUTE_NO_OVERWRITE', unfinished)
    return lock

if __name__ == '__main__':
    print(json.dumps({'preflight_pass': bool(verify()), 'gpu_operations': 0}))
