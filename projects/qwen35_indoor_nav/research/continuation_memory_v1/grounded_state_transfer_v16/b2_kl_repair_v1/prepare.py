"""Bind exact original data/cache/schedules and sealed control weights; no new features."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *


def copy_bound(source,target,ledger):
    expected=sha(source);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected:raise ValueError('BOUND_COPY_CHANGED:'+str(target))
    else:shutil.copyfile(source,target)
    ledger[str(target.relative_to(target_run))]=dict(source=str(source),sha256=expected)


def main(run):
    global target_run
    target_run=run;verify_lock(read(run/'SOURCE_LOCK.json'));cfg=read(run/'PROTOCOL.json');bindings={}
    if read(CONTROL/'RESULT.json')['complete']!=1152:raise ValueError('CONTROL_NOT_COMPLETE')
    for name in ('DATA.json','DATA_BINDING.json','FIT_AUXILIARY_WEIGHTS.json','features/FEATURES.pt','features/FEATURE_RESULT.json'):
        copy_bound(CONTROL/name,run/name,bindings)
    # Retain preprocessing fingerprints and cache state seals, not old navigation trajectories.
    for source in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        if not (source.parent/'STATE_SEAL.json').exists():continue
        for p in (source,source.parent/'STATE_SEAL.json'):
            copy_bound(p,run/p.relative_to(CONTROL),bindings)
    for seed in cfg['seeds']:
        copy_bound(CONTROL/'train'/f'SCHEDULE_{seed}.json',run/'train'/f'SCHEDULE_{seed}.json',bindings)
        for arm in ('B1','B2'):
            source=CONTROL/'train'/f'{arm}_{seed}'
            for name in ('FINAL.pt','RESULT.json'):copy_bound(source/name,run/'train'/source.name/name,bindings)
            result=read(source/'RESULT.json')
            if result['updates']!=1200 or sha(source/'FINAL.pt')!=result['checkpoint_sha256']:raise ValueError('CONTROL_WEIGHT_IDENTITY')
    from evaluate_continuations import registry_value
    reg=registry_value(read(run/'DATA.json')['raw_families'],cfg)
    if reg['conditions']!=read(CONTROL/'EVALUATION_REGISTRY.json')['conditions']:raise ValueError('CONDITIONS_CHANGED')
    immutable(run/'EVALUATION_REGISTRY.json',reg)
    immutable(run/'REUSE_BINDING.json',dict(files=bindings,control_protocol_sha256=sha(CONTROL/'PROTOCOL.json'),control_source_lock_sha256=sha(CONTROL/'SOURCE_LOCK.json'),old_navigation_reused=False))

if __name__=='__main__':main(Path(sys.argv[1]))
