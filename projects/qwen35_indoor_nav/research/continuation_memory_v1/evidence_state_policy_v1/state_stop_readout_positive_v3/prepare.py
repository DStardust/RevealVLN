"""Bind the same six weight files and DEV conditions, with a new inference operator."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run);source=Path(cfg['source_run']);verify_lock(read(source/'SOURCE_LOCK.json'))
    for path,expected in read(source/'BINDING.json')['files'].items():
        if sha(Path(path))!=expected:raise ValueError('SOURCE_DATA_CHANGED')
    files={source/name:run/name for name in ('DATA.json','WARMUP_DATA.json','features/FEATURES.pt','features/FEATURE_RESULT.json')}
    for p in (source/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        files[p]=run/p.relative_to(source);files[p.parent/'STATE_SEAL.json']=run/p.parent.relative_to(source)/'STATE_SEAL.json'
    receipts=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            tag=f'{arm}_{seed}';source_tag=tag.replace('STOPPLUS','STOPFIX');folder=source/'train'/source_tag;record=read(folder/'RESULT.json')
            if record['updates']!=1200 or sha(folder/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('FIXED_CHECKPOINT_CHANGED')
            for name in ('FINAL.pt','RESULT.json'):files[folder/name]=run/'train'/tag/name
            receipts.append(dict(model=tag,source_model=source_tag,path=str((folder/'FINAL.pt').relative_to(LINE)),sha256=record['checkpoint_sha256'],state_sha256=record['final']))
    for src,dst in files.items():
        dst.parent.mkdir(exist_ok=True,parents=True)
        if dst.exists():
            if sha(dst)!=sha(src):raise ValueError('COPY_CHANGED')
        else:shutil.copyfile(src,dst)
    from evaluate_continuations import registry_value
    reg=registry_value(read(run/'DATA.json')['raw_families'],cfg)
    if len(reg['slots'])!=768 or reg['conditions']!=read(source/'EVALUATION_REGISTRY.json')['conditions']:raise ValueError('DEV_CONDITIONS_CHANGED')
    immutable(run/'EVALUATION_REGISTRY.json',reg)
    immutable(run/'DATA_AUDIT.json',dict(data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json'),source_run=str(source),same_original_dev_conditions=True))
    immutable(run/'MODEL_REUSE.json',dict(models=receipts,new_updates=0,original_model_updates=0,selection='All fixed three-seed finals; STOPPLUS uses identical STOPFIX tensors with nonnegative residual operator'))
    immutable(run/'BINDING.json',dict(files={str(p):sha(p) for p in files.values()},source_run=str(source),source_result_sha256=sha(source/'RESULT.json')))

if __name__=='__main__':main(Path(sys.argv[1]))
