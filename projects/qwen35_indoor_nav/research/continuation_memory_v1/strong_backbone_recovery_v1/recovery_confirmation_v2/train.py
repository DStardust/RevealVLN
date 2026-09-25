"""Shared demonstrations, full causal unroll, fixed final checkpoint, no unseen loss."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
import numpy as np
import torch
import torch.nn.functional as F
from confirmation_model import ConfirmationMemory, sequence_logits
from preserve_objective_v3 import preservation_loss
from transfer_pipeline import records


def objective(model,recovery,ordinary,weights):
    r=sequence_logits(model,recovery);mask=recovery['known']
    recovery_ce=F.cross_entropy(r[mask],recovery['targets'][mask],weight=weights)
    o=sequence_logits(model,ordinary)
    keep,kl,margin=preservation_loss(o,ordinary['base_logits'],ordinary['targets'])
    ordinary_ce=F.cross_entropy(o,ordinary['targets'])
    loss=recovery_ce+.5*ordinary_ce+5*keep
    return loss,dict(recovery_ce=float(recovery_ce.detach()),ordinary_ce=float(ordinary_ce.detach()),
        ordinary_kl=float(kl.detach()),ordinary_margin=float(margin.detach()),
        recovery_queries=int(mask.sum()),ordinary_queries=len(o),
        recovery_accuracy=float((r[mask].argmax(-1)==recovery['targets'][mask]).float().mean()),
        ordinary_agreement=float((o.argmax(-1)==ordinary['targets']).float().mean()))


def main(run,arm):
    u.verify_sources(run);p=u.read(run/'PROTOCOL.json');entry=p['models'][arm];seed=entry['seed'];data=run/'data'/str(seed);a=u.read(data/'ADMISSION.json')
    for name,key in [('POOLS.pt','pools_sha256'),('SCHEDULE.json','schedule_sha256'),('INITIAL.pt','initial_sha256')]:
        assert u.sha(Path(a['pools_path']) if name=='POOLS.pt' else data/name)==a[key]
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ConfirmationMemory(3584,entry['architecture']);initial=torch.load(data/'INITIAL.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(initial);model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    folder=run/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    binding=dict(arm=arm,protocol_sha256=u.sha(run/'PROTOCOL.json'),source_sha256=u.sha(run/'SOURCE_LOCK.json'),**a)
    saved=sorted(folder.glob('checkpoint_*.pt'));start=0
    if saved:
        c=torch.load(saved[-1],map_location='cpu',weights_only=False);assert c['binding']==binding
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);start=c['step']
        torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng'])
    if (folder/'FINAL.pt').exists():
        c=torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=False);assert c['binding']==binding and c['step']==p['steps']
        model.load_state_dict(c['model']);start=p['steps']
    pack=torch.load(Path(a['pools_path']),map_location='cpu',weights_only=True)
    # Only FIT rows move into the training pool. DEV is loaded for diagnostics after final step.
    rows={i:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in row.items()}
          for i,row in enumerate(pack['rows']) if row['partition']=='FIT'}
    weights=pack['class_weights'].cuda();schedule=u.read(data/'SCHEDULE.json');began=time.time()
    u.write(folder/'RUNTIME_IDENTITY.json',dict(arm=arm,gpu=torch.cuda.get_device_name(0),dtype='float32_memory/float32_generation_scores',
        base_loaded=False,base_updates=0,parameter_count=sum(x.numel() for x in model.parameters()),
        architecture=entry['architecture'],seed=seed,attention_parameters_unused=True,recurrent_unused=entry['architecture']=='LOCAL',initial_sha256=a['initial_sha256']))
    with (folder/f'STEPS_{time.time_ns()}.jsonl').open('x',buffering=1) as log:
        for step in range(start,p['steps']):
            ri,oi=schedule[step];loss,info=objective(model,rows[ri],rows[oi],weights)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()),'NONFINITE_TRAINING'
            writer_grad=float(model.writer.weight.grad.norm())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['clip_grad_norm']);optimizer.step()
            info.update(step=step+1,planned=p['steps'],arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),writer_gradient_norm=writer_grad,
                schedule=[ri,oi],wall_seconds=time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(info,allow_nan=False)+'\n')
            if (step+1)%10==0:u.write(folder/'PROGRESS.json',dict(status='TRAINING',**info))
            if (step+1)%p['checkpoint_every']==0:
                path=folder/f'checkpoint_{step+1:06d}.pt';tmp=path.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,binding=binding,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),tmp);tmp.replace(path)
    assert any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()),'NO_UPDATE'
    final=folder/'FINAL.pt';tmp=folder/'FINAL.tmp'
    if not final.exists():
        torch.save(dict(model=model.state_dict(),arm=arm,architecture=entry['architecture'],seed=seed,step=p['steps'],binding=binding),tmp);tmp.replace(final)
    diagnostic=[];model.eval()
    with torch.no_grad():
        for row in pack['rows']:
            if row['partition']!='DEV':continue
            d={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in row.items()}
            logits=sequence_logits(model,d);known=d['known']
            diagnostic.append(dict(id=d['id'],kind=d['kind'],queries=int(known.sum()),
                correct=int((logits.argmax(-1)[known]==d['targets'][known]).sum()),
                native_correct=int((d['base_logits'].argmax(-1)[known]==d['targets'][known]).sum())))
    u.write(folder/'RESULT.json',dict(status='COMPLETE',actual_steps=p['steps'],base_updates=0,head_changed=True,
        final_sha256=u.sha(final),dev_action_diagnostic=diagnostic,dev_used_for_selection=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();main(a.run,a.arm)
