"""Exclusive CPU freeze after tests; not runtime approval or job preparation."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def main():
    if (HERE/'SHA256SUMS').exists() or (HERE/'SOURCE_LOCK.json').exists():raise ValueError('NO_SEAL_OVERWRITE')
    files=sorted(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md']
    with (HERE/'SHA256SUMS').open('x') as f:
        f.write(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
    s=importlib.util.spec_from_file_location('scale_scout_seal_prepare',HERE/'prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    lock={str(p):h for p,h in m.code_dependencies().items()}
    for p in [m.c.QUEUE,m.c.QUEUE.parent/'SOURCE_LOCK.json']:lock[str(p)]=m.c.sha(p)
    m.c.save(HERE/'SOURCE_LOCK.json',lock)
    value=dict(status='CPU_READY_FOR_MAIN_REVIEW_NOT_RUNTIME_APPROVED',cpu_tests=25,source_hashes=len(lock),
        source_queue=str(m.c.QUEUE),source_queue_sha256=m.c.sha(m.c.QUEUE),prospective_houses=43,prospective_positions=172,
        jobs_prepared=0,gpu_operations=0,new_physical_hubs=0,new_physical_families=0,scientific_pass=False,
        runtime_allowed=False,holder_api_sealed=True,transport_seconds=3300,audit_seconds=1800,max_seconds=5100,
        source_lock_sha256=m.c.sha(HERE/'SOURCE_LOCK.json'))
    m.c.save(HERE/'result.json',value);print(json.dumps(value,indent=2))
if __name__=='__main__':main()
