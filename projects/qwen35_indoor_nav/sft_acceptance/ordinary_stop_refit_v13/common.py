"""Versioned ordinary STOP-row training and paired navigation evidence."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
ROOT=LINE.parents[1]
ASSET=Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav')
OLD=ASSET/'sft_acceptance/ordinary_stop_row_v12'
V5=LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
V16=LINE/'research/continuation_memory_v1/grounded_state_transfer_v16'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
c=load('stop13_frozen_io',V5/'common.py')
read,write,append,sha=c.read,c.write,c.append,c.sha
def verify():
    for path,h in read(HERE/'SOURCE_LOCK.json')['files'].items():
        if sha(Path(path))!=h:raise ValueError('SOURCE_OR_ASSET_CHANGED:'+path)
def tensor_hash(t):return c.tensor_identity(t)
def validate_head(original,candidate):
    import torch
    assert set(original)==set(candidate),'PARAMETER_KEYS_CHANGED'
    for name,t in original.items():
        if name in ('action_head.weight','action_head.bias'):
            assert torch.equal(t[:3],candidate[name][:3]),'MOTION_ROWS_CHANGED'
        else:assert torch.equal(t,candidate[name]),'NONSTOP_PARAMETER_CHANGED'
        assert torch.isfinite(candidate[name]).all(),'NONFINITE_PARAMETERS'
def state_stamp(state):
    values={k:tensor_hash(v) for k,v in state.items()}
    return dict(tensors=values,sha256=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest())
