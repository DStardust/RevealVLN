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
from recovery_model import RecoveryMemory, sequence_logits
from preserve_objective_v3 import preservation_loss
from transfer_pipeline import records


def prepare(run):
    p=u.read(run/'PROTOCOL.json');pool=[];counts=Counter();certificates={}
    rows=records(run/'features','evaluation',True)
    expected={e['id'] for e in u.read(run/'features/DATA_MANIFEST.json')['episodes']}
    assert set(rows)==expected,'INCOMPLETE_FEATURES'
    for index,g in sorted(rows.items()):
        c=g['cache'];assert u.sha(c['path'])==c['sha256'];certificates[c['path']]=c['sha256']
        row=torch.load(c['path'],map_location='cpu',weights_only=True)
        row['id']=index
        assert len(row['targets'])==len(row['actor_features'])==len(row['known'])==len(row['query_steps'])
        assert row['known'].any() and row['memory_features'].shape[1]==3584
        if row['partition']=='FIT' and row['kind']=='RECOVERY':counts.update(row['targets'][row['known']].tolist())
        pool.append(row)
    fit_houses={r['house'] for r in pool if r['partition']=='FIT'}
    assert not fit_houses & {r['house'] for r in pool if r['partition']=='DEV'}
    rec=[i for i,r in enumerate(pool) if r['partition']=='FIT' and r['kind']=='RECOVERY']
    ordinary=[i for i,r in enumerate(pool) if r['partition']=='FIT' and r['kind']=='PRESERVATION']
    assert rec and ordinary and all(counts[a]>0 for a in range(4)),'RECOVERY_ACTION_COVERAGE_MISSING'
    weights=torch.tensor([1/(counts[a]**.5) for a in range(4)]);weights/=weights.mean()
    out=run/'data';out.mkdir(exist_ok=True)
    if (out/'POOLS.pt').exists():raise ValueError('DATA_ALREADY_PREPARED')
    torch.save(dict(rows=pool,class_weights=weights),out/'POOLS.pt')
    rng=random.Random(p['seed']);schedule=[]
    for step in range(p['steps']):
        if step%len(rec)==0:rng.shuffle(rec)
        if step%len(ordinary)==0:rng.shuffle(ordinary)
        schedule.append([rec[step%len(rec)],ordinary[step%len(ordinary)]])
    u.write(out/'SCHEDULE.json',schedule)
    torch.manual_seed(p['seed']);initial=RecoveryMemory(3584,'CONCAT')
    torch.save(initial.state_dict(),out/'INITIAL.pt')
    u.write(out/'ADMISSION.json',dict(counts=dict(Counter(r['partition']+':'+r['kind'] for r in pool)),
        action_counts=dict(counts),class_weights=weights.tolist(),certificates=certificates,
        pools_sha256=u.sha(out/'POOLS.pt'),schedule_sha256=u.sha(out/'SCHEDULE.json'),initial_sha256=u.sha(out/'INITIAL.pt'),
        no_old_heads_loaded=True,no_unseen_labels=True,full_unroll=True,base_updates=0))


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
    u.verify_sources(run);p=u.read(run/'PROTOCOL.json');a=u.read(run/'data/ADMISSION.json')
    for name,key in [('POOLS.pt','pools_sha256'),('SCHEDULE.json','schedule_sha256'),('INITIAL.pt','initial_sha256')]:
        assert u.sha(run/'data'/name)==a[key]
    torch.set_num_threads(4);random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    model=RecoveryMemory(3584,arm);initial=torch.load(run/'data/INITIAL.pt',map_location='cpu',weights_only=True)
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
    pack=torch.load(run/'data/POOLS.pt',map_location='cpu',weights_only=True)
    # Only FIT rows move into the training pool. DEV is loaded for diagnostics after final step.
    rows={i:{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in row.items()}
          for i,row in enumerate(pack['rows']) if row['partition']=='FIT'}
    weights=pack['class_weights'].cuda();schedule=u.read(run/'data/SCHEDULE.json');began=time.time()
    u.write(folder/'RUNTIME_IDENTITY.json',dict(arm=arm,gpu=torch.cuda.get_device_name(0),dtype='float32_memory/float32_generation_scores',
        base_loaded=False,base_updates=0,parameter_count=sum(x.numel() for x in model.parameters()),
        concat_unused_attention_parameters=arm=='CONCAT',initial_sha256=a['initial_sha256']))
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
        torch.save(dict(model=model.state_dict(),arm=arm,step=p['steps'],binding=binding),tmp);tmp.replace(final)
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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['CONCAT','EVIDENCE']);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.run) if a.prepare else main(a.run,a.arm)
