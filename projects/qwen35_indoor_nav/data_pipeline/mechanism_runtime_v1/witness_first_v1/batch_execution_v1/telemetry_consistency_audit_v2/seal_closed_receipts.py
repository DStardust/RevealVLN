"""Exclusive manifests for the two completed independent 06r2 audit outputs."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parents[1]
TARGETS=(HERE/'batch06r2_closed_v1',WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/independent_03_04_07_v1/batch_06r2')

if __name__=='__main__':
    for target in TARGETS:
        assert target==target.resolve() and target.is_relative_to(WF)
        paths=sorted(p for p in target.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
        assert paths and all(p==p.resolve() for p in paths)
        records=[(hashlib.sha256(p.read_bytes()).hexdigest(),p.relative_to(target).as_posix()) for p in paths]
        with (target/'SHA256SUMS').open('x') as stream:
            stream.write(''.join(f'{sha}  {name}\n' for sha,name in records))
        assert all(hashlib.sha256((target/name).read_bytes()).hexdigest()==sha for sha,name in records)
        print(target,len(records),'VERIFIED')
