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

if __name__=='__main__':main(Path(sys.argv[1]))
