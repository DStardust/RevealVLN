"""Freeze transport only; no approvals and no GPU launch."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport as t


def main():
    t.verify_lock(t.NEXT/'INPUT_LOCK.json')
    paths=list(HERE.glob('*.py'))+[HERE/'README.md',t.NEXT/'INPUT_LOCK.json']
    for i in (0,1):
        path=HERE/f'TRANSPORT_DRAFT_SHARD_{i}.json'
        t.save(path,dict(t.transport_record(i),executable=False,runtime_allowed=False))
        paths.append(path)
    t.save(HERE/'INPUT_LOCK.json',{str(p.relative_to(t.ROOT)):t.sha(p) for p in paths})
    print(json.dumps(dict(status='CPU_READY_FOR_MAIN_REVIEW',input_lock_sha256=t.sha(HERE/'INPUT_LOCK.json'),
        locked_files=len(paths),executable=False,gpu_operations=0)))


if __name__=='__main__':main()
