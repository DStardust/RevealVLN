"""Exclusive source freeze; does not launch or query GPUs."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    files=sorted(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md']
    with (HERE/'SHA256SUMS').open('x') as f:
        f.write(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
    print('SEALED',len(files))
