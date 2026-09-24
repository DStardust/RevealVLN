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
c=load('survival17_frozen_io',V5/'common.py')
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

_gt={}
def groundtruth(path):
    import gzip
    if path not in _gt:_gt[path]=json.load(gzip.open(path))
    return _gt[path]

def audit_episode(folder,index,gt_path,collection=False):
    """Frozen public metrics, parameterized GT split, and actual action transport."""
    import math
    e=read(folder/f'episode_{index:02d}.json');policy=c.records(folder/'POLICY_STEPS.jsonl');steps=c.records(folder/'STEPS_PRIVILEGED.jsonl')
    assert len(policy)==len(steps)==e['steps']<=500,'TRACE_LENGTH'
    assert e['stopped'] or e['steps']==500 or (collection and e['termination']=='TEACHER_UNAVAILABLE'),'NONSTOP_EARLY_END'
    assert len(e['positions'])==len(e['distances'])==e['steps']+1,'PATH_LENGTH'
    interface=c.records(folder/'INTERFACE.jsonl')[0];previous=interface['rgb_sha256'];last=None;executed=[]
    for i,(a,s) in enumerate(zip(policy,steps),1):
        expected=[previous] if i==1 else [last,previous]
        assert a['step']==s['step']==i and a['executed_action']==s['action'],'ACTION_TRANSPORT'
        assert a['raw']['executed_history']==executed[-8:] and a['raw']['rgb_sha256']==expected,'HISTORY_TRANSPORT'
        assert a['raw']['instruction']==e['instruction'],'INSTRUCTION_CHANGED'
        if not collection or e.get('mode')=='POLICY':
            assert c.ACTIONS[max(range(4),key=lambda j:a['logits'][j])]==a['executed_action'],'METHOD_ARGMAX'
        assert s['action']!='STOP' or i==len(steps),'POST_STOP_ACTION'
        assert s['position']==e['positions'][i] and s['distance_to_goal']==e['distances'][i],'METRIC_TRACE_ALIGNMENT'
        executed.append(s['action']);last,previous=previous,s['rgb_sha256']
    sr=float(e['stopped'] and e['distances'][-1]<3);length=sum(math.dist(a,b) for a,b in zip(e['positions'],e['positions'][1:]));spl=sr*e['distances'][0]/max(e['distances'][0],length)
    aggregate=load('survival17_frozen_metric_audit',LINE/'closed_loop_bench/ordinary_cycle_pair_gpu1_v3/aggregate.py')
    ndtw=aggregate.exact_ndtw_check(e['positions'],groundtruth(gt_path)[str(e['episode_id'])]['locations'])
    assert sr==e['success'] and e['oracle_success']==float(min(e['distances'])<3),'SUCCESS_DEFINITION'
    assert math.isclose(spl,e['spl'],rel_tol=1e-5,abs_tol=1e-5) and math.isclose(ndtw,e['ndtw'],rel_tol=1e-9,abs_tol=1e-9),'METRIC_RECOMPUTATION'
    return e
