"""Frozen unused-hub selection across source snapshots, never outcome selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ROOT=WF.parents[4]
def sha(p):
    p=Path(p).resolve(strict=True);assert p.is_relative_to(ROOT)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def main(old,new,name):
    old,new=Path(old).resolve(strict=True),Path(new).resolve(strict=True)
    assert old.is_relative_to(WF) and new.is_relative_to(WF)
    lock={};rows=[];selection=[]
    for source,indices in [(old,[3,4]),(new,[0])]:
        draft=json.loads((source/'CONFIG_DRAFT.json').read_text())
        assert not draft['runtime_allowed'] and not draft['training_allowed']
        for p,h in json.loads((source/'SOURCE_LOCK.json').read_text()).items():
            assert sha(p)==h,p
            assert p not in lock or lock[p]==h,p
            lock[p]=h
        lock[str(source/'CONFIG_DRAFT.json')]=sha(source/'CONFIG_DRAFT.json')
        rows.extend(draft['candidates'][i] for i in indices)
        selection.append({'snapshot':str(source),'indices':indices})
    prior=json.loads((HERE/'batch_00/run_v1/EXECUTION_CONFIG.json').read_text())['candidates']
    for i,row in enumerate(rows):
        for other in prior+rows[:i]:
            assert row['house_id']!=other['house_id'] or math.dist(row['configuration']['u_position'],other['configuration']['u_position'])>=1,'HUB_OVERLAP'
    out=HERE/name;out.mkdir(exist_ok=False)
    cfg=dict(node='Q35N_FROZEN_UNUSED_HUB_COMPOSITE',runtime_allowed=False,executable=False,training_allowed=False,
             candidates=rows,factory_variant='winding_v1',source_selection=selection,
             selection_rule='old source remaining one-per-hub indices 3/4 plus first new-source candidate; no runtime outcome used')
    lock[str(Path(__file__).resolve())]=sha(__file__)
    for filename,obj in [('CONFIG_DRAFT.json',cfg),('SOURCE_LOCK.json',lock)]:
        with (out/filename).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
    print(json.dumps(dict(prepared=True,source=str(out),houses=[r['house_id'] for r in rows])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--name',required=True)
    a=p.parse_args();assert '/' not in a.name and a.name.startswith('composite_');main(a.old,a.new,a.name)
