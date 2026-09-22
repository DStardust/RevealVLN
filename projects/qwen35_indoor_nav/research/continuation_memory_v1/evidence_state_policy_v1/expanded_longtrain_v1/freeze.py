"""Freeze this version and inherited read-only runtime sources."""
from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parent;C=D.parent/'data_scale_validation_v1';LINE=D.parent.parents[2];ROOT=LINE.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        while b:=s.read(8*1024**2):h.update(b)
    return h.hexdigest()
old=json.loads((C/'SOURCE_LOCK.json').read_text());files=old['files'].copy()
for rel,h in files.items():
    if sha(LINE/rel)!=h:raise ValueError('INHERITED_SOURCE_CHANGED:'+rel)
for p in list(D.glob('*.py'))+[D/'PROTOCOL.json',D/'index.html']:files[str(p.relative_to(LINE))]=sha(p)
out=dict(files=files,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),origin=str(C/'SOURCE_LOCK.json'),origin_sha256=sha(C/'SOURCE_LOCK.json'))
with (D/'SOURCE_LOCK.json').open('x') as s:json.dump(out,s,indent=2);s.write('\n')
print('locked',len(files))
