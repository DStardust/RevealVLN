"""Freeze actual sources and raw-data identities before execution, without touching old evidence."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent;LINE=B.parents[2];ROOT=LINE.parents[1];V=B.parent/'grounded_state_transfer_v16'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  while x:=s.read(8*1024**2):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
paths={};origins={}
for source in [B/'monotonic_holdout_v1/SOURCE_LOCK.json',B/'event_data_scale20_v2/SOURCE_LOCK.json']:
 for rel,expected in read(source)['files'].items():
  p=LINE/rel
  if sha(p)!=expected:raise ValueError('INHERITED_SOURCE_CHANGED:'+rel)
  paths[rel]=expected
 origins[str(source.relative_to(LINE))]=sha(source)
for folder in (D,B,V):
 for p in folder.glob('*.py'):paths[str(p.relative_to(LINE))]=sha(p)
for p in (B/'event_data_scale20_v2/quality.py',B/'event_data_scale20_v2/disk_watch_r2.py',D/'PROTOCOL.json',B/'runs/cpu_003/DATA.json',B/'runs/cpu_003/SCHEDULES.json',B/'event_data_scale20_v2/runs/supplement_001/DATASET.json',B/'monotonic_holdout_v1/runs/holdout_001/DATA.json'):
 paths[str(p.relative_to(LINE))]=sha(p)
lock=dict(files=paths,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),origins=origins,scope='Only new version sources plus actually read frozen dependencies/data. Old results unchanged.')
p=D/'SOURCE_LOCK.json'
if p.exists():raise FileExistsError('SOURCE_LOCK already frozen')
p.write_text(json.dumps(lock,indent=2)+'\n');print('locked',len(paths))
