"""CPU-only fixed probes, reporting every seed on both held-house partitions."""
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare_screen
c=prepare_screen.c
witness=prepare_screen.witness


def main(feature_run):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_ONLY'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config=c.read(HERE/'SCREEN_PROTOCOL.json')
    feature_config=c.read(HERE/'FEATURE_PROTOCOL.json')
    assert c.sha(Path(__file__))==feature_config['source_hashes'][str(Path(__file__).relative_to(prepare_screen.LINE))]
    receipt=c.read(feature_run/'FEATURE_RESULT.json')
    assert c.read(feature_run/'LAUNCH_RESULT.json')['status']=='COMPLETE'
    assert receipt['parameters_unchanged'] and receipt['optimizer_updates']==0
    assert receipt['screen_sha256']==c.sha(HERE/'SCREEN.json')
    assert receipt['file_sha256']==c.sha(feature_run/'FEATURES.pt')
    data=c.read(HERE/'SCREEN.json')
    run=HERE/'screen_cpu_run_001';run.mkdir(exist_ok=False)
    began=time.monotonic()
    features=torch.load(feature_run/'FEATURES.pt',map_location='cpu',weights_only=True)['features'].float()
    rows=data['rows']
    x=features[[r['feature'] for r in rows]]
    y=torch.tensor([r['y'] for r in rows],dtype=torch.float32)
    mask=torch.tensor([r['mask'] for r in rows],dtype=torch.bool)
    fit=torch.tensor([r['split']=='fit' for r in rows])
    mean=x[fit].mean(0);scale=x[fit].std(0,unbiased=False).clamp_min(1e-4)
    x=(x-mean)/scale
    weights=torch.zeros_like(y)
    for family in sorted({r['family_id'] for r in rows if r['split']=='fit'}):
        family_mask=torch.tensor([r['family_id']==family for r in rows])
        for target in range(2):
            for label in (0,1):
                index=family_mask & mask[:,target] & (y[:,target]==label)
                if index.any():weights[index,target]=1/int(index.sum())
    weights *= int(fit.sum())/weights.sum(0)
    fit_indices=fit.nonzero().flatten()
    results={}
    for seed in config['seeds']:
        generator=torch.Generator().manual_seed(seed)
        schedule=torch.randint(len(fit_indices),(config['steps'],config['batch_size']),generator=generator)
        for name in config['heads']:
            torch.manual_seed(seed)
            head=nn.Linear(2048,2) if name=='linear' else nn.Sequential(nn.Linear(2048,128),nn.Tanh(),nn.Linear(128,2))
            initial=c.model_identity(head)['sha256']
            optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            first_gradient=None
            for step,batch in enumerate(schedule,1):
                chosen=fit_indices[batch]
                optimizer.zero_grad(set_to_none=True)
                loss=(F.binary_cross_entropy_with_logits(head(x[chosen]),y[chosen],reduction='none')*weights[chosen]).mean()
                assert bool(torch.isfinite(loss));loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
                if first_gradient is None:
                    first_gradient=sum(float(p.grad.square().sum()) for p in head.parameters())**.5
                    assert first_gradient>0
                optimizer.step()
                if step==1 or step%100==0:
                    c.append(run/'STEPS.jsonl',dict(seed=seed,head=name,step=step,loss=float(loss.detach())))
                    c.write(run/'PROGRESS.json',dict(unix=time.time(),seed=seed,head=name,step=step))
            final=c.model_identity(head)['sha256'];assert final!=initial
            with torch.no_grad():
                logits=torch.cat([head(chunk) for chunk in x.split(512)])
            key=f'{name}_{seed}'
            torch.save(dict(head=head.state_dict(),mean=mean,scale=scale),run/f'{key}.pt')
            c.write(run/f'{key}_PREDICTIONS.json',logits.tolist(),True)
            def grouped(field):
                return {value:witness.confusion(logits,y,mask,torch.tensor([r[field]==value for r in rows]))
                        for value in sorted({r[field] for r in rows})}
            result=dict(partitions=grouped('partition'),houses=grouped('house'),families=grouped('family_id'),
                initial_state_sha256=initial,final_state_sha256=final,first_gradient_norm=first_gradient,
                optimizer_updates=config['steps'],parameters_updated=True)
            results[key]=result
            c.write(run/f'{key}_RESULT.json',result,True)
            print(key,result['partitions'],flush=True)
    c.write(run/'RESULT.json',dict(status='CAUSAL_WITNESS_TRANSFER_SCREEN_COMPLETE',results=results,
        seconds=time.monotonic()-began,diagnostic_head_updates=len(results)*config['steps'],
        runtime_policy_updates=0,gpu_hours=0,
        feature_receipt_sha256=c.sha(feature_run/'FEATURE_RESULT.json'),
        screen_protocol_sha256=c.sha(HERE/'SCREEN_PROTOCOL.json'),
        feature_protocol_sha256=c.sha(HERE/'FEATURE_PROTOCOL.json'),
        original_training_admission=False,closed_loop_benefit_measured=False,publication_ready=False,
        interpretation='All fixed heads and seeds; balanced current-witness screen only. Additional houses are held from lightweight training, not assumed unseen to the base. Labels/queries never enter the encoder.'),True)


if __name__=='__main__':
    main(Path(sys.argv[1]))
