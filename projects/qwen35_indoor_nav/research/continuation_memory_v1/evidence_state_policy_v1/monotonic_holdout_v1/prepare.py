"""Bind six completed heads and old warm-up data without training or selecting weights."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run);source=Path(cfg['source_models']);verify_lock(read(source/'SOURCE_LOCK.json'))
    files={CPU/'DATA.json':run/'WARMUP_DATA.json',CONTROL/'features/FEATURES.pt':run/'features/FEATURES.pt',
           CONTROL/'features/FEATURE_RESULT.json':run/'features/FEATURE_RESULT.json'}
    for path in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        files[path]=run/path.relative_to(CONTROL);files[path.parent/'STATE_SEAL.json']=run/path.parent.relative_to(CONTROL)/'STATE_SEAL.json'
    receipts=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            tag=f'{arm}_{seed}';record=read(source/'train'/tag/'RESULT.json');path=source/'train'/tag/'FINAL.pt'
            if record['updates']!=1200 or record['base_updates'] or sha(path)!=record['checkpoint_sha256']:raise ValueError('FROZEN_FINAL_HEAD')
            for name in ('FINAL.pt','RESULT.json'):files[source/'train'/tag/name]=run/'train'/tag/name
            receipts.append(dict(model=tag,path=str(path),sha256=record['checkpoint_sha256'],state_sha256=record['final']))
    for source,target in files.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(source)!=sha(target):raise ValueError('BOUND_COPY_CHANGED:'+str(target))
        else:shutil.copyfile(source,target)
    immutable(run/'MODEL_REUSE.json',dict(models=receipts,new_optimizer_updates=0,base_updates=0))
    immutable(run/'BINDING.json',dict(files={str(p):sha(p) for p in files.values()},house_manifest_sha256=sha(HERE/'HOUSE_MANIFEST.json')))

if __name__=='__main__':main(Path(sys.argv[1]))
