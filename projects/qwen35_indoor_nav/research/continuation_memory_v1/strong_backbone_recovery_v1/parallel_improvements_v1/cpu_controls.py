"""Bounded CPU implementation check on two real FIT traces; no efficacy test."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'recovery_confirmation_v2'),str(HERE.parent)]
import common as u
import torch
from control_models import UnitReadoutMemory
from confirmation_model import sequence_logits
spec=importlib.util.spec_from_file_location('frozen_confirmation_train',HERE.parent/'recovery_confirmation_v2/train.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
objective=training.objective


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1209)
    began=time.time()
    output=HERE/'CPU_CONTROLS_RESULT.json'
    if output.exists():
        raise ValueError('CLOSED_IMPLEMENTATION_CHECK')
    source=HERE.parent/'recovery_action_v1/runs/action_001/data'
    admission=u.read(source/'ADMISSION.json')
    assert u.sha(source/'POOLS.pt')==admission['pools_sha256']
    pool=torch.load(source/'POOLS.pt',map_location='cpu',weights_only=True,mmap=True)
    # Fixed first id within each FIT kind, not selected by model performance.
    rows={kind:min((r for r in pool['rows'] if r['partition']=='FIT' and r['kind']==kind),key=lambda r:r['id'])
          for kind in ('RECOVERY','PRESERVATION')}
    reference=UnitReadoutMemory(3584,'RECURRENT')
    initial=copy.deepcopy(reference.state_dict())
    results={}
    for mode in ('LOCAL','EMA','RECURRENT'):
        head=UnitReadoutMemory(3584,mode)
        head.load_state_dict(initial)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
        updates=[]
        for step in range(3):
            loss,info=objective(head,rows['RECOVERY'],rows['PRESERVATION'],pool['class_weights'])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in head.parameters())
            writer_gradient=float(head.writer.weight.grad.norm())
            recurrent_gradient=float(head.recurrent.weight.grad.norm()) if head.recurrent.weight.grad is not None else 0.
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
            updates.append(dict(step=step+1,loss=float(loss.detach()),writer_gradient=writer_gradient,recurrent_gradient=recurrent_gradient))
            if time.time()-began>600:
                raise RuntimeError('CPU_IMPLEMENTATION_BUDGET')
        assert updates[-1]['writer_gradient']>0
        assert (updates[-1]['recurrent_gradient']>0)==(mode=='RECURRENT')
        changed=[k for k,v in head.state_dict().items() if not torch.equal(v,initial[k])]
        assert 'writer.weight' in changed and 'actor.2.weight' in changed
        results[mode]=dict(parameter_count=sum(v.numel() for v in head.parameters()),updates=updates,changed_parameters=changed)
    result=dict(status='CPU_IMPLEMENTATION_CHECK_COMPLETE',gpu_hours=0,base_loaded=False,base_updates=0,
        actual_lightweight_updates=9,rows={k:dict(id=r['id'],house=r['house'],partition=r['partition'],
            sequence_length=len(r['memory_features']),known_queries=int(r['known'].sum())) for k,r in rows.items()},
        pool_sha256=admission['pools_sha256'],models=results,cpu_seconds=time.time()-began,
        same_initial_state=True,full_causal_unroll=True,dev_or_unseen_used=False,
        efficacy_measured=False,claim='Working strong-simple-control interfaces only. No navigation benefit or new algorithmic result.',
        final_weights_retained=False)
    u.write(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models',)}))


if __name__=='__main__':
    main()
