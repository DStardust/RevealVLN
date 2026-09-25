"""Preservation diagnostics, complete paired denominators and nonzero-task gate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
from transfer_pipeline import records


def summarize(run):
    manifest=u.read(run/'DATA_MANIFEST.json')['episodes'];planned={r['id'] for r in manifest}
    rows=records(run,'evaluation',True);assert set(rows)<=planned
    protocol=u.read(run/'PROTOCOL.json');arms={};pairs={};costs={}
    for row in rows.values():
        group=Path(row['path']).parent;session=group.parents[1]
        seal=u.read(session/'STATE_SEAL.json')
        assert seal['base_before']==seal['base_after']==protocol['expected_base_state_sha256']
        assert seal['heads_unchanged'] and u.sha(session/'RUNTIME_IDENTITY.json')==row['runtime_identity_sha256']
        identity=u.read(session/'RUNTIME_IDENTITY.json');assert identity['heads']==protocol['heads']
        for arm,path in row['trace_hashes'].items():assert u.sha(group/arm/'TRACE.jsonl')==path
    for arm in ('NATIVE','BC','B2','OURS'):
        values=[r['outcomes'][arm] for r in rows.values()];successes=sum(bool(v['success']) for v in values)
        n=len(values);den=len(planned)
        arms[arm]=dict(complete=n,planned=den,successes=successes,sr_completed=successes/n if n else None,
            sr_full=successes/den if n==den else None,identification_bounds=[successes/den,(successes+den-n)/den],
            spl=sum(v['spl'] for v in values)/n if n else None,mean_steps=sum(v['steps'] for v in values)/n if n else None)
        early=far=overrides=0
        for row in rows.values():
            trace=[json.loads(s) for s in (Path(row['path']).parent/arm/'TRACE.jsonl').read_text().splitlines()]
            actions=[r for r in trace if r['event']=='action'];generation=[r for r in trace if r['event']=='generation']
            early+=bool(actions and len(actions)<=5 and actions[-1]['executed_action']==0)
            far+=bool(actions and actions[-1]['executed_action']==0 and actions[-1]['distance']>=3)
            overrides+=sum(r['native_token']!=r['method_token'] for r in generation)
        costs[arm]=dict(stop_within_five=early,far_stop=far,actual_action_token_overrides=overrides)
        if arm!='NATIVE':
            wins=[i for i,r in rows.items() if r['outcomes'][arm]['success']>r['outcomes']['NATIVE']['success']]
            losses=[i for i,r in rows.items() if r['outcomes'][arm]['success']<r['outcomes']['NATIVE']['success']]
            pairs[arm]=dict(wins=wins,losses=losses,delta_sr=(len(wins)-len(losses))/den if n==den else None)
    houses={h:{a:dict(n=sum(r['house']==h for r in rows.values()),successes=sum(bool(r['outcomes'][a]['success']) for r in rows.values() if r['house']==h)) for a in arms} for h in sorted({x['house'] for x in manifest})}
    return dict(planned_groups=len(planned),complete_groups=len(rows),missing=sorted(planned-set(rows)),
        evaluation_complete=set(rows)==planned,arms=arms,paired=pairs,by_house=houses,costs=costs,
        meaning='Ordinary SR preservation; FIT action correction is not recovery generalization',full_1839=False)


def gradient_audit(root):
    import torch
    from memory_v2 import ExecutionMemory
    from preserve_objective_v3 import sequence_logits,preservation_loss
    torch.set_num_threads(4);p=u.read(root/'PROTOCOL.json')
    pool=torch.load(root/'data/POOLS.pt',map_location='cpu',weights_only=True)
    row=pool['ordinary_fit'][0];model=ExecutionMemory(row['memory_features'].shape[-1])
    old=Path(p['memory_run'])/'training/OURS/FINAL.pt'
    model.load_state_dict(torch.load(old,map_location='cpu',weights_only=False)['model'])
    logits=sequence_logits(model,row);loss,kl,margin=preservation_loss(logits,row['base_logits'],row['targets'])
    loss.backward();norm=sum(float(x.grad.square().sum()) for x in model.parameters() if x.grad is not None)**.5
    stop_gradient=float(model.actor[-1].bias.grad[0]);assert len(logits)>0 and kl>0 and norm>0
    assert stop_gradient>0,'PRESERVATION_DOES_NOT_REDUCE_OLD_STOP_COLLAPSE'
    u.write(root/'PRESERVATION_GRADIENT.json',dict(status='MEASURED_NONEMPTY_LOSS_AND_GRADIENT',queries=len(logits),
        episode_id=row['episode_id'],partition='FIT',loss=float(loss),kl=float(kl),margin=float(margin),gradient_norm=norm,
        stop_bias_gradient=stop_gradient,diagnostic_head_sha256=u.sha(old),training_uses_fresh_initialization=True,
        scope='Backward-only test on old collapsed head; not a new training result or SR'))


def training(root,trial):
    import torch
    from memory_v2 import ExecutionMemory
    from preserve_objective_v3 import sequence_logits
    from review_memory_r1 import review_split
    torch.set_num_threads(4);p=u.read(root/'PROTOCOL.json');folder=root/'trials'/str(trial)
    pool=torch.load(root/'data/POOLS.pt',map_location='cpu',weights_only=True)
    sparse=torch.load(Path(p['memory_run'])/'data/FIT.pt',map_location='cpu',weights_only=True)
    states={};ordinary={};receipt={};schedule=u.read(folder/'training/SCHEDULE.json')
    for arm in ('BC','B2','OURS'):
        path=folder/'training'/arm/'FINAL.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        assert u.sha(path)==u.read(path.parent/'RESULT.json')['final_sha256']
        assert saved['step']==1200 and saved['binding']['base_updates']==0
        journal=[json.loads(s) for s in (path.parent/'STEPS.jsonl').read_text().splitlines()];steps={s['step']:s for s in journal}
        assert set(steps)==set(range(1,1201)) and [steps[s]['schedule'] for s in range(1,1201)]==schedule
        assert all(r['ordinary_queries']>0 and r['dense_queries']>0 for r in journal)
        states[arm]=saved['model'];model=ExecutionMemory(pool['ordinary_dev'][0]['memory_features'].shape[-1]).eval();model.load_state_dict(states[arm])
        correct=total=0;changes=[]
        with torch.inference_mode():
            for row in pool['ordinary_dev']:
                pred=sequence_logits(model,row).argmax(-1);truth=row['targets'];count=int((pred==truth).sum())
                correct+=count;total+=len(truth);changes.append(dict(episode_id=row['episode_id'],correct=count,total=len(truth)))
        ordinary[arm]=dict(correct=correct,total=total,action_agreement=correct/total,episodes=changes)
        receipt[arm]=dict(actual_updates_including_replayed=len(journal),weight_sha256=u.sha(path),
            ordinary_supervised_queries=sum(r['ordinary_queries'] for r in journal),last_ordinary_kl=steps[1200]['ordinary_preservation_kl'])
    fit=review_split(sparse,states)
    # Keep useful nonzero task behavior: a zero head scores zero; constant STOP scores 8/10.
    eligible=[r['arm'] for r in fit['arms'] if r['arm'] in ('B2','OURS') and r['correct']==r['total']==10]
    u.write(folder/'TRAINING_REVIEW.json',dict(fit=fit,ordinary_dev=ordinary,training=receipt,
        retained_task_arms=eligible,scope='Exposed FIT fit and ordinary DEV behavior, not recovery generalization'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['gradient','training'],required=True)
    p.add_argument('--trial',type=int,default=0);a=p.parse_args();u.verify_sources(a.root)
    gradient_audit(a.root) if a.phase=='gradient' else training(a.root,a.trial)
