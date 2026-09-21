"""Reuse all six exact checkpoints, original DEV registry and frozen causal cache."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run);source=Path(cfg['source_run']);verify_lock(read(source/'SOURCE_LOCK.json'))
    for path,expected in read(source/'BINDING.json')['files'].items():
        if sha(Path(path))!=expected:raise ValueError('SOURCE_DATA_CHANGED')
    files={source/name:run/name for name in ('DATA.json','WARMUP_DATA.json','EVALUATION_REGISTRY.json','DATA_AUDIT.json','features/FEATURES.pt','features/FEATURE_RESULT.json','SEMANTIC_STOP_AUDIT.json')}
    for p in (source/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        files[p]=run/p.relative_to(source);files[p.parent/'STATE_SEAL.json']=run/p.parent.relative_to(source)/'STATE_SEAL.json'
    receipts=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            tag=f'{arm}_{seed}';folder=source/'train'/tag;record=read(folder/'RESULT.json')
            if record['updates']!=1200 or sha(folder/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('FIXED_CHECKPOINT_CHANGED')
            for name in ('FINAL.pt','RESULT.json'):files[folder/name]=run/'train'/tag/name
            receipts.append(dict(model=tag,path=str((folder/'FINAL.pt').relative_to(LINE)),sha256=record['checkpoint_sha256'],state_sha256=record['final']))
    for src,dst in files.items():
        dst.parent.mkdir(exist_ok=True,parents=True)
        if dst.exists():
            if sha(dst)!=sha(src):raise ValueError('COPY_CHANGED')
        else:shutil.copyfile(src,dst)
    from evaluate_continuations import registry
    reg=registry(run)
    if len(reg['slots'])!=768:raise ValueError('DEV_DENOMINATOR')
    immutable(run/'MODEL_REUSE.json',dict(models=receipts,new_updates=0,original_model_updates=0,selection='All fixed three-seed finals, no model selection'))
    immutable(run/'BINDING.json',dict(files={str(p):sha(p) for p in files.values()},source_run=str(source),source_result_sha256=sha(source/'RESULT.json')))

if __name__=='__main__':main(Path(sys.argv[1]))
