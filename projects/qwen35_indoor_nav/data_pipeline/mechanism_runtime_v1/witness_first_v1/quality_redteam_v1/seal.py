"""Seal this closed TEST_FIXTURE node only; never edits source production files."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
target=HERE/'SHA256SUMS'
assert not target.exists()
rows=[]
for p in sorted(HERE.rglob('*')):
    if p.is_file():
        assert not p.is_symlink()
        rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(HERE))+'\n')
with target.open('x') as f:f.writelines(rows)
print('sealed_files',len(rows))
