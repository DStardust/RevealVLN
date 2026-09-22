"""Versioned GPU wiring; CPU model, data and losses remain read-only."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
LINE=PARENT.parents[2]
ROOT=LINE.parents[1]
V16=PARENT.parent/'grounded_state_transfer_v16'
PILOT=V16/'identifiable_pilot_v1'
CONTROL=V16/'b2_matched_control_v1/runs/matched_001'
CPU=PARENT/'runs/cpu_003'
sys.path.append(str(V16))
from v16_common import c,read,write,append,sha,digest,immutable,atomic_torch,load,DATA_LINE

def verify_lock(lock):
    for path,expected in lock['files'].items():
        if sha(LINE/path)!=expected:raise ValueError('SOURCE_CHANGED:'+path)

def config(run):
    value=read(run/'PROTOCOL.json');verify_lock(read(run/'SOURCE_LOCK.json'));return value

def runtime_config(run):
    import os
    value=read(run/'PROTOCOL.json')
    if 'B2_DEVICE' in os.environ:value.update(read(Path(os.environ['B2_DEVICE'])))
    return value

def make_head(tag):
    mode,seed=tag.rsplit('_',1)
    if 'evidence_gpu_architecture' not in sys.modules:
        common=load('evidence_gpu_cpu_common',PARENT/'common.py')
        sys.modules['common']=common
        load('evidence_gpu_architecture',PARENT/'model.py')
    sys.path.insert(0,str(HERE))
    return sys.modules['evidence_gpu_architecture'].EvidencePolicy(int(seed),"MONOTONIC")

def prefill(net,features):
    import torch
    device=next(net.parameters()).device
    state=net.reset(1,device)
    for feature in features:
        _,state,_=net.step(feature,torch.zeros(1,4,device=device),state)
    return state

def state_identity(state):
    import torch
    return {name:c.tensor_identity(value.clone(memory_format=torch.contiguous_format))
            for name,value in state._asdict().items()}
