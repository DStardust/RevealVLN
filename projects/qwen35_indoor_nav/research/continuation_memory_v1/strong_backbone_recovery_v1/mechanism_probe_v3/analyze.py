"""CPU-only memory readout ablation on fixed recorded DEV teacher trajectories."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/'recovery_confirmation_v2'),str(HERE.parent)]
import common as u
import torch
import torch.nn.functional as F
from confirmation_model import ConfirmationMemory


def compare(model,row):
    lookup={int(t):q for q,t in enumerate(row['query_steps'])};memory=model.reset();actual=[];without=[]
    last=max(lookup);features=row['memory_features'][:last+1].detach().clone().requires_grad_(True)
    for t,x in enumerate(features):
        memory=model.update(x[None],memory)
        if t in lookup:
            q=lookup[t];actor=row['actor_features'][q:q+1]
            actual.append(model.action_delta(actor,memory)[0])
            without.append(model.action_delta(actor,torch.zeros_like(memory))[0])
    full=row['base_logits'].float()+torch.stack(actual)
    zero=row['base_logits'].float()+torch.stack(without)
    # A readout ablation on this recorded trajectory, not a counterfactual rollout.
    loss=F.cross_entropy(full[-1:],row['targets'][-1:]);loss.backward()
    magnitude=features.grad.norm(dim=-1);old=magnitude[:-8].sum() if len(magnitude)>8 else magnitude.new_zeros(())
    return full.detach(),zero.detach(),dict(last_query=last,old_input_gradient_mass=float(old),
        total_input_gradient_mass=float(magnitude.sum()),old_gradient_fraction=float(old/magnitude.sum().clamp_min(1e-20)),
        gradient_scope='Final-query CE through only the added branch; actor features held fixed; sensitivity, not causal necessity')


def main(source,out):
    torch.set_num_threads(2);began=time.time();out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():raise ValueError('CLOSED_DIAGNOSTIC')
    p=u.read(source/'PROTOCOL.json');a=u.read(source/'data/42/ADMISSION.json');pool=Path(a['pools_path'])
    assert u.sha(pool)==a['pools_sha256']
    rows=[r for r in torch.load(pool,map_location='cpu',weights_only=True)['rows'] if r['partition']=='DEV']
    assert len(rows)==64
    u.write(out/'PROTOCOL.json',dict(source_run=str(source),source_protocol_sha256=u.sha(source/'PROTOCOL.json'),
        pool_sha256=a['pools_sha256'],models=p['models'],rows=64,planned_model_rows=384,gpu_use=False,
        supervised_scope='Recorded DEV teacher recovery and native preservation trajectories; not autonomous unseen',
        intervention='Set added readout memory to zero, keep recorded actor features/base logits/input sequence fixed',
        limitation='Zeroing may be out of distribution; neither feature accuracy nor input gradients imply navigation SR gains',
        selection='All six registered models and all64 DEV rows; no hyperparameter/weight updates'))
    summaries={};complete=0
    for name,entry in p['models'].items():
        checkpoint=source/'training'/name/'FINAL.pt';receipt=u.read(checkpoint.parent/'RESULT.json')
        assert u.sha(checkpoint)==receipt['final_sha256']
        model=ConfirmationMemory(3584,entry['architecture']);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['arm']==name and saved['step']==3000
        model.load_state_dict(saved['model']);model.requires_grad_(False);model.eval()
        model_dir=out/name;model_dir.mkdir(exist_ok=True)
        for row in rows:
            target=model_dir/f'{row["id"]}.json'
            if target.exists():complete+=1;continue
            if time.time()-began>1800:raise RuntimeError('CPU_BUDGET_REACHED_CAN_RESUME')
            full,zero,gradient=compare(model,row);known=row['known'];labels=row['targets']
            native=row['base_logits'].argmax(-1);f=full.argmax(-1);z=zero.argmax(-1)
            assert torch.isfinite(full).all() and torch.isfinite(zero).all()
            record=dict(id=row['id'],house=row['house'],kind=row['kind'],checkpoint_sha256=receipt['final_sha256'],
                known_queries=int(known.sum()),full_correct=int(((f==labels)&known).sum()),zero_correct=int(((z==labels)&known).sum()),
                native_correct=int(((native==labels)&known).sum()),changed_queries=int(((f!=z)&known).sum()),
                full_helped=int(((f==labels)&(z!=labels)&known).sum()),full_hurt=int(((f!=labels)&(z==labels)&known).sum()),
                known_ce_full=float(F.cross_entropy(full[known],labels[known])),known_ce_zero=float(F.cross_entropy(zero[known],labels[known])),
                all_query_steps=row['query_steps'].tolist(),known_mask=known.tolist(),targets=labels.tolist(),
                full_actions=f.tolist(),zero_actions=z.tolist(),native_actions=native.tolist(),gradient=gradient)
            u.write(target,record);complete+=1
            u.write(out/'STATUS.json',dict(status='RUNNING',complete_model_rows=complete,planned_model_rows=384,model=name,
                cpu_wall_seconds=time.time()-began,gpu_hours=0))
        values=[u.read(model_dir/f'{r["id"]}.json') for r in rows];summaries[name]={}
        for kind in ('RECOVERY','PRESERVATION'):
            group=[v for v in values if v['kind']==kind]
            counts={k:sum(v[k] for v in group) for k in ['known_queries','full_correct','zero_correct','native_correct','changed_queries','full_helped','full_hurt']}
            counts['mean_old_gradient_fraction']=sum(v['gradient']['old_gradient_fraction'] for v in group)/len(group)
            summaries[name][kind]=counts
        u.write(out/'PARTIAL_RESULT.json',dict(models=summaries,complete_model_rows=complete))
    u.write(out/'RESULT.json',dict(status='COMPLETE',models=summaries,complete_model_rows=complete,planned_model_rows=384,
        gpu_hours=0,optimizer_updates=0,cpu_wall_seconds=time.time()-began,navigation_sr_measured=False))
    u.write(out/'STATUS.json',dict(status='COMPLETE',complete_model_rows=complete,planned_model_rows=384,gpu_hours=0))
    lines=['COMPLETE','','固定已记录的 DEV 教师恢复/原生成功轨迹，六个模型；不是自主导航 SR。',
        'full/zero 使用同一 actor 特征和原生分数。零记忆可能离开训练分布；梯度表示敏感性，不能证明记忆必要或长程语义正确。','']
    for name,groups in summaries.items():
        for kind,v in groups.items():lines.append(f"{name} {kind}: full {v['full_correct']}/{v['known_queries']}, zero {v['zero_correct']}/{v['known_queries']}; memory帮助/损害标签预测 {v['full_helped']}/{v['full_hurt']}")
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a.source,a.output)
    except BaseException as e:
        u.write(a.output/'FAILURE.json',dict(error=str(e),gpu_hours=0));raise
