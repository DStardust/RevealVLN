"""Read-only CPU validation of the first new checkpoint, not navigation evidence."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_expanded_low_lr_v3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_ACCEPTANCE'
    import torch
    p=read(TRAIN/'PROTOCOL_FILESTORE.json')
    for name,digest in p['code_sha256'].items():assert sha(TRAIN/name)==digest,name
    old=Path(p['resume_from']['path']);assert sha(old)==p['resume_from']['sha256']
    checkpoint=TRAIN/'formal/attempt_001/checkpoint_000004200.pt';receipt=read(str(checkpoint)+'.json')
    assert sha(checkpoint)==receipt['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    before=torch.load(old,map_location='cpu',weights_only=True)
    assert state['cursor']==receipt['cursor'] and state['cursor']['updates']==state['cursor']['position']==4200
    assert state['cursor']['epoch']==0 and state['cursor']['decisions']>before['cursor']['decisions']
    assert state['binding']['protocol_sha256']==sha(TRAIN/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==p['sample_index_sha256']
    assert state['global_decisions']==state['charged_compute_decisions']==receipt['global_decisions']>340712
    assert set(state['trainable'])==set(before['trainable'])
    changed=[]
    for k,v in state['trainable'].items():
        assert torch.isfinite(v).all()
        if not torch.equal(v,before['trainable'][k]):changed.append(k)
    assert changed and len(state['optimizer']['state'])==len(before['optimizer']['state'])
    for slot in state['optimizer']['state'].values():
        assert int(slot['step'])==4200,'OPTIMIZER_RESTARTED'
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    total=31059;warmup=int(.03*total);expected_lr=5e-6*.5*(1+math.cos(math.pi*(4199-warmup)/(total-warmup)))
    assert all(math.isclose(group['lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15) for group in state['optimizer']['param_groups'])
    history=[json.loads(x) for x in (TRAIN/'formal/attempt_001/PROGRESS.jsonl').read_text().splitlines() if x.strip()]
    first=[x for x in history if x['cursor']['updates']<=4200]
    assert first and all(4000<x['cursor']['updates']<=4200 and x['resume_updates']==4000 for x in first)
    assert all(math.isfinite(x['metrics']['mean_ce']) and math.isfinite(x['metrics']['grad_norm']) and x['metrics']['grad_norm']>0 for x in first)
    resume=[]
    for line in (TRAIN/'formal/attempt_001/train.log').read_text().splitlines():
        if line.startswith('{'):
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('event')=='RESUME_READY':resume.append(r)
    assert len(resume)==3 and {x['rank'] for x in resume}=={0,1,2}
    counters=read(TRAIN/'RESUME_AUDIT.json')['rank_decisions_before']
    assert all(x['cursor']['updates']==4000 and x['cursor']['decisions']==counters[x['rank']] and x['charged_base']==340712 for x in resume)
    locks=LINE/'data_pipeline/ordinary_expansion_v1'
    assert sha(locks/'runtime_v2/INPUT_LOCK.json')=='c693ff433c3a02557c5d125400ce77d877b396a4d26544ffd7420fba1189509f'
    assert sha(locks/'runtime_gpu3_recovery_v1/INPUT_LOCK.json')=='fcb9eb183ccf48b66e6a532b55c23550b96c4c76312cdc42a309bae73aedf6f4'
    result=dict(status='PASS',unix=time.time(),accepted_checkpoint=str(checkpoint),checkpoint_sha256=receipt['sha256'],
        updates=4200,new_updates=200,new_decisions=state['global_decisions']-340712,
        optimizer_step_preserved=True,optimizer_moments_not_reset=True,learning_rate=expected_lr,
        parameter_tensors_changed=len(changed),all_tensors_finite=True,rank_resume_counters=counters,
        source_and_old_input_locks_unchanged=True,navigation_gain_verified=False)
    with (HERE/'FIRST_CHECKPOINT_ACCEPTANCE.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
