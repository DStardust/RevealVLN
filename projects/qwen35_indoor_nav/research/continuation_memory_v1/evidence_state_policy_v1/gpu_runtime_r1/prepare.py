"""Bind current CPU-admitted labels, original caches and pre-registered conditions."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run)
    if read(CPU/'RESULT.json')['status']!='READY_FOR_GPU_STAGE_NOT_LAUNCHED':raise ValueError('CPU_NOT_READY')
    binding=read(CPU/'BINDING.json')
    for path,value in binding['files'].items():
        if sha(Path(path))!=value:raise ValueError('CPU_ADMISSION_CHANGED:'+path)
    copies={CPU/'DATA.json':run/'DATA.json',CPU/'SCHEDULES.json':run/'SCHEDULES.json',
            CONTROL/'features/FEATURES.pt':run/'features/FEATURES.pt',
            CONTROL/'features/FEATURE_RESULT.json':run/'features/FEATURE_RESULT.json'}
    for source in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        for p in (source,source.parent/'STATE_SEAL.json'):copies[p]=run/p.relative_to(CONTROL)
    for source,target in copies.items():
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=sha(source):raise ValueError('BOUND_COPY_CHANGED')
        else:shutil.copyfile(source,target)
    binding['files'].update({str(p.resolve()):sha(p) for p in (run/'DATA.json',run/'SCHEDULES.json',run/'PROTOCOL.json')})
    binding['files'].update({str((LINE/p).resolve()):value for p,value in read(run/'SOURCE_LOCK.json')['files'].items()})
    immutable(run/'BINDING.json',binding)
    from evaluate_continuations import registry_value
    reg=registry_value(read(run/'DATA.json')['raw_families'],cfg)
    if reg['conditions']!=read(CONTROL/'EVALUATION_REGISTRY.json')['conditions']:raise ValueError('CONDITIONS_CHANGED')
    immutable(run/'EVALUATION_REGISTRY.json',reg)
    (run/'train').mkdir(exist_ok=True)
    schedules=read(run/'SCHEDULES.json')
    for seed in cfg['seeds']:immutable(run/'train'/f'SCHEDULE_{seed}.json',schedules[str(seed)])
    immutable(run/'DATA_BINDING.json',dict(data_sha256=sha(run/'DATA.json'),source_cpu_run=str(CPU),new_data=0))
    original=Path(cfg['training_reused_from'])
    verify_lock(read(original/'SOURCE_LOCK.json'))
    for name in ('DATA.json','SCHEDULES.json','EVALUATION_REGISTRY.json'):
        if sha(run/name)!=sha(original/name):raise ValueError('REPAIR_CHANGED_EXPERIMENT:'+name)
    expected=[f'{a}_{seed}' for seed in cfg['seeds'] for a in cfg['arms']]
    if read(original/'DIAGNOSTICS_COMPLETE.json')['models']!=expected:raise ValueError('INCOMPLETE_SOURCE_MODELS')
    receipts=[]
    for tag in expected:
        record=read(original/'train'/tag/'RESULT.json')
        if record['updates']!=1200 or record['base_updates']!=0:raise ValueError('UNFINISHED_MODEL')
        if sha(original/'train'/tag/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('CHANGED_MODEL')
        for name in ('FINAL.pt','RESULT.json','FIT_WEIGHTS.json','PROGRESS.json'):
            source=original/'train'/tag/name;target=run/'train'/tag/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=sha(source):raise ValueError('REUSED_HEAD_CHANGED')
            else:shutil.copyfile(source,target)
        immutable(run/('DIAG_'+tag+'.json'),read(original/('DIAG_'+tag+'.json')))
        receipts.append(dict(model=tag,sha256=record['checkpoint_sha256'],updates=record['updates']))
    immutable(run/'DIAGNOSTICS_COMPLETE.json',read(original/'DIAGNOSTICS_COMPLETE.json'))
    immutable(run/'TRAINING_REUSE.json',dict(source=str(original),models=receipts,new_updates=0,
        cumulative_experiment_updates=10800,old_evaluation_groups=0,
        reason='CONTENT_PATH failed before first reset; preserve trained weights and repair only storage scope'))
    # Carry the original resource ledger forward once, including failed evaluation cost.
    if not (run/'PRIOR_RESOURCES.json').exists():
        for row in c.records(original/'RESOURCES.jsonl'):append(run/'RESOURCES.jsonl',dict(row,inherited_from=str(original)))
        immutable(run/'PRIOR_RESOURCES.json',dict(source=str(original/'RESOURCES.jsonl'),sha256=sha(original/'RESOURCES.jsonl')))


if __name__=='__main__':main(Path(sys.argv[1]))
