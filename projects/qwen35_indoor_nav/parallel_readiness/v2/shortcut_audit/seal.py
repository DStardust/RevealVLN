"""Seal only this audit directory; no shared writes."""
import hashlib
from pathlib import Path

out=Path(__file__).resolve().parent
entries=[]
for p in sorted(out.iterdir()):
    if p.is_file() and p.name!='SHA256SUMS':
        entries.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
(out/'SHA256SUMS').write_text('\n'.join(entries)+'\n')
print({'sealed_files':len(entries)})
