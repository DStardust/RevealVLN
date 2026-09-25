"""Fit only real TRAIN continuation gain/harm; never consume unseen labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
import torch
import torch.nn.functional as F
from intervention_gate_v1 import InterventionGate,policy_features,label,choose_method
from selection import disagreement
from transfer_pipeline import records
from transfer_runtime import tensor_hash


def gather(root):
    run=root/'train';p=u.read(run/'PROTOCOL.json');u.verify_sources(run)
    manifest={e['id']:e for e in u.read(run/'DATA_MANIFEST.json')['episodes']}
    groups=records(run,'evaluation',True)
    assert set(groups)==set(manifest),'INCOMPLETE_TRAIN_COLLECTION'
    rows=[];unchanged=[]
    for index,group in groups.items():
        folder=Path(group['path']).parent;session=folder.parents[1]
        seal=u.read(session/'STATE_SEAL.json');identity=u.read(session/'RUNTIME_IDENTITY.json')
        assert seal['base_before']==seal['base_after']==p['expected_base_state_sha256'] and seal['heads_unchanged']
        assert identity['heads']==p['heads']
        for arm in ('BC','B2','OURS'):
            trace=folder/arm/'TRACE.jsonl';assert u.sha(trace)==group['trace_hashes'][arm]
            queries=[json.loads(s) for s in trace.read_text().splitlines() if json.loads(s)['event']=='generation']
            changes=[(i,r) for i,r in enumerate(queries) if disagreement(r)]
            episode=manifest[index]
            meta=dict(id=index,source='new_recovery_v2',house=group['house'],partition=episode['partition'],variant=episode['variant'],arm=arm,
                native_success=bool(group['outcomes']['NATIVE']['success']),method_success=bool(group['outcomes'][arm]['success']),
                native_steps=group['outcomes']['NATIVE']['steps'],method_steps=group['outcomes'][arm]['steps'])
            if not changes:
                assert group['audits'][arm]['full_trajectory_matched'];unchanged.append(meta);continue
            q,first=changes[0];audit=group['audits'][arm]
            assert audit['input_prefix_matched'] and audit['action_prefix_matched']
            assert first['environment_step']==audit['first_executed_override_step']
            cache=folder/arm/'FEATURES.pt';features=torch.load(cache,map_location='cpu',weights_only=True)['features']
            assert len(features)==len(queries)
            actor=features[q].clone();assert tensor_hash(actor)==first['feature_sha256']
            rows.append(dict(actor=actor,native=torch.tensor(first['native_logits']),method=torch.tensor(first['method_logits']),
                target=label(meta['native_success'],meta['method_success']),trace_path=str(trace),trace_sha256=group['trace_hashes'][arm],
                first_step=first['environment_step'],feature_sha256=u.sha(cache),**meta))
    old=Path(p['old_fit_source']);previous=old/'gate_data/PAIRS.pt'
    assert u.sha(previous)==u.read(old/'gate_data/COUNTS.json')['pairs_sha256']
    old_pack=torch.load(previous,map_location='cpu',weights_only=True)
    assert u.read(old/'PROTOCOL.json')['heads']==p['heads']
    for r in old_pack['rows']:rows.append(dict(r,source='calibrate_001',variant='natural'))
    for r in old_pack['unchanged']:unchanged.append(dict(r,source='calibrate_001',variant='natural'))
    fit={r['house'] for r in rows+unchanged if r['partition']=='FIT'}
    dev={r['house'] for r in rows+unchanged if r['partition']=='DEV'}
    assert not fit&dev,'HOUSE_LEAKAGE'
    out=root/'gate_data';out.mkdir(exist_ok=True);torch.save(dict(rows=rows,unchanged=unchanged),out/'PAIRS.pt')
    counts={a:{s:dict(Counter(['NEUTRAL','GAIN','HARM'][r['target']] for r in rows if r['arm']==a and r['partition']==s)) for s in ('FIT','DEV')} for a in ('BC','B2','OURS')}
    u.write(out/'COUNTS.json',dict(by_arm_partition=counts,complete_new_groups=len(groups),reused_old_groups=200,
        intervention_examples=len(rows),unchanged_pairs=len(unchanged),pairs_sha256=u.sha(out/'PAIRS.pt'),
        actual_label='Whole remaining frozen policy outcome, not every changed action; forced shared turns are not gate examples'))
    return rows,unchanged


def main(root):
    torch.set_num_threads(4);u.verify_sources(root/'train')
    if (root/'GATE_REVIEW.json').exists():return
    rows,unchanged=gather(root);p=u.read(root/'train/PROTOCOL.json');results={}
    for arm in ('BC','B2','OURS'):
        fit=[r for r in rows if r['arm']==arm and r['partition']=='FIT']
        dev=[r for r in rows if r['arm']==arm and r['partition']=='DEV']
        counts=Counter(r['target'] for r in fit)
        if not counts[1] or not counts[2]:
            results[arm]=dict(status='MISSING_GAIN_OR_HARM',counts=dict(counts),actual_steps=0);continue
        folder=root/'gates'/arm;folder.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(42);gate=InterventionGate(3592);optimizer=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.1)
        x=torch.stack([policy_features(r['actor'],r['native'],r['method']) for r in fit]);y=torch.tensor([r['target'] for r in fit])
        weight=torch.tensor([len(fit)/(len(counts)*counts[c]) if counts[c] else 0 for c in range(3)])
        final=folder/'FINAL.pt';attempt=time.time_ns()
        if final.exists():gate.load_state_dict(torch.load(final,map_location='cpu',weights_only=True)['model'])
        else:
            with (folder/f'STEPS_{attempt}.jsonl').open('x') as log:
                for step in range(p['gate']['steps']):
                    loss=F.cross_entropy(gate.linear(x),y,weight=weight);optimizer.zero_grad(set_to_none=True);loss.backward()
                    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in gate.parameters()),'NONFINITE_GRADIENT'
                    norm=torch.nn.utils.clip_grad_norm_(gate.parameters(),1);optimizer.step()
                    if (step+1)%20==0:
                        row=dict(arm=arm,step=step+1,planned=p['gate']['steps'],loss=float(loss.detach()),gradient_norm=float(norm),fit_counts=dict(counts))
                        log.write(json.dumps(row)+'\n');log.flush();u.write(folder/'PROGRESS.json',row)
            assert all(torch.isfinite(v).all() for v in gate.state_dict().values()),'NONFINITE_FINAL_GATE'
            torch.save(dict(model=gate.state_dict(),arm=arm,steps=p['gate']['steps'],counts=dict(counts),class_weights=weight,
                source_lock_sha256=u.sha(root/'train/SOURCE_LOCK.json'),data_sha256=u.sha(root/'gate_data/PAIRS.pt')),
                folder/'FINAL.tmp')
            (folder/'FINAL.tmp').replace(final)
        with torch.no_grad():
            accepted=choose_method(gate.linear(torch.stack([policy_features(r['actor'],r['native'],r['method']) for r in dev]))).tolist() if dev else []
        fixed=[r for r in unchanged if r['arm']==arm and r['partition']=='DEV']
        n=len(dev)+len(fixed);assert n==200
        native=sum(r['native_success'] for r in dev+fixed)
        ungated=sum(r['method_success'] for r in dev+fixed)
        selected=sum(r['method_success'] if accept else r['native_success'] for r,accept in zip(dev,accepted))+sum(r['native_success'] for r in fixed)
        decisions=[dict(id=r['id'],source=r['source'],house=r['house'],variant=r['variant'],target=r['target'],accepted=a) for r,a in zip(dev,accepted)]
        u.write(folder/'DEV_BRANCH_SELECTION.json',dict(decisions=decisions,not_live_rollout=True,not_unseen=True))
        results[arm]=dict(status='TRAINED',actual_steps=p['gate']['steps'],counts=dict(counts),path=str(final),sha256=u.sha(final),
            dev_planned=n,dev_native=native,dev_ungated=ungated,dev_selected_logged_branches=selected,
            dev_used_for_selection=False,scope='DEV logged branch diagnostic only; live unseen evaluation follows regardless of score')
    u.write(root/'GATE_REVIEW.json',dict(status='GATES_READY' if any(r['status']=='TRAINED' for r in results.values()) else 'DATA_LIMITED',
        results=results,actual_total_steps=sum(r['actual_steps'] for r in results.values()),unseen_used_for_training=False,
        EU6_action_anchors_used=False,new_architecture_trained=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run)
