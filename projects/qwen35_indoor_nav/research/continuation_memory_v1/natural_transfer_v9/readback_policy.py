"""CPU-only saved-weight readback, including the deployed native-STOP protection.

All decisions remain teacher-forced observations; this is not a rollout metric.
"""
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    run=HERE/'run_001';config=c.read(HERE/'PROTOCOL.json');data=c.read(HERE/'DATA.json')
    identity=c.read(run/'FEATURE_RESULT.json')
    assert c.sha(run/'FEATURES.pt')==identity['file_sha256']
    cache={k:v.float() for k,v in torch.load(run/'FEATURES.pt',map_location='cpu',weights_only=True).items()}
    special=c.read(HERE.parent/'multifamily_v7/DATA.json')
    result=c.read(run/'RESULT.json')
    records=[r for r in data['records'] if r['partition']=='check']
    rows={}
    for seed in config['seeds']:
        for arm in config['arms']:
            key=f'{arm}_{seed}';path=run/f'{key}_MEMORY.pt'
            net=train.models.MemoryPolicy(2048,len(special['query_vocabulary']),8,64,.99).eval()
            net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            expected=result['runs'][key]['final_state_sha256']
            assert c.model_identity(net)['sha256']==expected
            counts=dict(decisions=0,native_correct=0,raw_head_correct=0,protected_choice_correct=0,
                native_stop_decisions=0,raw_head_attempted_stop_override=0,native_stop_with_motion_teacher=0,
                additional_stop_decisions=0)
            for start in range(0,len(records),8):
                selected=records[start:start+8];batch=train.ordinary_batch(selected,'cpu')
                _,logits=train.ordinary_loss(net,cache,batch)
                native=cache['logits'][batch['indices']].argmax(-1)
                raw=logits.argmax(-1);chosen=torch.where(native==3,native,raw)
                mask=batch['mask'];target=batch['targets']
                assert bool(torch.isfinite(logits).all())
                counts['decisions']+=int(mask.sum())
                counts['native_correct']+=int(((native==target)&mask).sum())
                counts['raw_head_correct']+=int(((raw==target)&mask).sum())
                counts['protected_choice_correct']+=int(((chosen==target)&mask).sum())
                counts['native_stop_decisions']+=int(((native==3)&mask).sum())
                counts['raw_head_attempted_stop_override']+=int(((native==3)&(raw!=3)&mask).sum())
                counts['native_stop_with_motion_teacher']+=int(((native==3)&(target!=3)&mask).sum())
                counts['additional_stop_decisions']+=int(((native!=3)&(chosen==3)&mask).sum())
            assert counts['decisions']==3769
            assert c.model_identity(net)['sha256']==expected
            rows[key]=dict(**counts,raw_head_accuracy=counts['raw_head_correct']/counts['decisions'],
                protected_choice_accuracy=counts['protected_choice_correct']/counts['decisions'],
                checkpoint_sha256=c.sha(path),state_sha256=expected,parameters_unchanged=True)
            c.write(HERE/'READBACK_PROGRESS.json',dict(unix=time.time(),completed=len(rows),key=key))
            print(key,counts,flush=True)
    c.write(HERE/'POLICY_READBACK.json',dict(status='ALL_FIXED_WEIGHTS_CPU_POLICY_READBACK',runs=rows,
        gpu_used=False,optimizer_updates=0,source_sha256=c.sha(Path(__file__)),
        protected_choice_rule='native STOP retained; otherwise argmax of memory-corrected four logits',
        scope='All60 CHECK teacher trajectories, not closed-loop rollouts; CPU memory forward over identical frozen causal features',
        checkpoint_selection=False),True)


if __name__=='__main__':main()
