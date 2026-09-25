"""Learn a single policy-switch decision from actual paired TRAIN rollouts.

The outcome target describes the complete remaining policy, not the causal
effect of every edited action. DEV reports are logged-branch selection, not a
new autonomous rollout. No unseen outcome enters fitting or gate selection.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from torch import nn
import torch.nn.functional as F
from transfer_pipeline import records
from memory_v2 import ExecutionMemory


def policy_features(actor,native,method):
    # Only the causal actor representation and the two current proposals.
    return torch.cat((F.layer_norm(actor,actor.shape[-1:]),(native-native.mean(-1,keepdim=True))/10,(method-native)),dim=-1)


class InterventionGate(nn.Module):
    def __init__(self,width):
        super().__init__();self.linear=nn.Linear(width,3)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,actor,native,method):return self.linear(policy_features(actor,native,method))


def choose_method(logits):
    # Classes: neutral / trajectory success gain / trajectory success harm.
    # Harm costs twice as much as gain; fixed before reading these outcomes.
    probability=logits.softmax(-1)
    return probability[...,1]>2*probability[...,2]


def label(native_success,method_success):
    return 1 if method_success>native_success else 2 if method_success<native_success else 0


def gather(run):
    manifest={r['id']:r for r in u.read(run/'DATA_MANIFEST.json')['episodes']};groups=records(run,'evaluation',True)
    assert set(groups)==set(manifest),'INCOMPLETE_PAIRED_COLLECTION'
    rows=[];unchanged=[]
    for index,group in groups.items():
        folder=Path(group['path']).parent;session=folder.parents[1];identity=u.read(session/'RUNTIME_IDENTITY.json')
        seal=u.read(session/'STATE_SEAL.json');assert seal['base_before']==seal['base_after']==u.read(run/'PROTOCOL.json')['expected_base_state_sha256']
        assert seal['heads_unchanged'] and identity['heads']==u.read(run/'PROTOCOL.json')['heads']
        for arm in ('BC','B2','OURS'):
            trace=folder/arm/'TRACE.jsonl';assert u.sha(trace)==group['trace_hashes'][arm]
            queries=[json.loads(s) for s in trace.read_text().splitlines() if json.loads(s)['event']=='generation']
            changed=[(i,r) for i,r in enumerate(queries) if r['native_token']!=r['method_token']]
            meta=dict(id=index,house=group['house'],partition=manifest[index]['partition'],arm=arm,
                native_success=bool(group['outcomes']['NATIVE']['success']),method_success=bool(group['outcomes'][arm]['success']),
                native_steps=group['outcomes']['NATIVE']['steps'],method_steps=group['outcomes'][arm]['steps'])
            if not changed:
                assert group['audits'][arm]['full_trajectory_matched'];unchanged.append(meta);continue
            q,first=changed[0]
            assert group['audits'][arm]['input_prefix_matched'] and group['audits'][arm]['action_prefix_matched']
            assert first['environment_step']==group['audits'][arm]['first_executed_override_step']
            cache=folder/arm/'FEATURES.pt';features=torch.load(cache,map_location='cpu',weights_only=True)['features']
            assert len(features)==len(queries)
            actor=features[q].clone();from transfer_runtime import tensor_hash
            assert tensor_hash(actor)==first['feature_sha256'],'ACTOR_FEATURE_NOT_THE_LOGGED_INPUT'
            rows.append(dict(actor=actor,native=torch.tensor(first['native_logits']),method=torch.tensor(first['method_logits']),
                target=label(meta['native_success'],meta['method_success']),first_step=first['environment_step'],
                trace_path=str(trace),trace_sha256=group['trace_hashes'][arm],feature_path=str(cache),feature_sha256=u.sha(cache),**meta))
    output=run/'gate_data';output.mkdir(exist_ok=True)
    torch.save(dict(rows=rows,unchanged=unchanged),output/'PAIRS.pt')
    counts={}
    for arm in ('BC','B2','OURS'):
        counts[arm]={split:dict(Counter(['NEUTRAL','GAIN','HARM'][r['target']] for r in rows if r['arm']==arm and r['partition']==split)) for split in ('FIT','DEV')}
    u.write(output/'COUNTS.json',dict(planned_physical_episodes=4*len(manifest),complete_groups=len(groups),
        first_intervention_examples=len(rows),unchanged_pairs=len(unchanged),by_arm_partition=counts,
        physical_unit='One complete NATIVE+BC+B2+OURS group per distinct route',
        targets='Whole remaining policy success difference, not individual-action correctness',
        pairs_sha256=u.sha(output/'PAIRS.pt')))
    return rows,unchanged


@torch.no_grad()
def task_anchors(protocol,arm):
    pack=torch.load(protocol['task_fit_pack'],map_location='cpu',weights_only=True)
    head=ExecutionMemory(3584).eval();head.load_state_dict(torch.load(protocol['heads'][arm]['path'],map_location='cpu',weights_only=False)['model'])
    anchors=[]
    for g in pack['groups']:
        x=g['features'];nt,nh,ns,w=x.shape
        delta=head(x.reshape(nt*nh,ns,w),g['lengths'].flatten(),g['actor_features'].reshape(nt*nh,ns,w))['delta'].reshape(nt,nh,ns,4)
        native=g['base_action_logits'];method=native+delta;mask=g['action_known']
        assert bool((method.argmax(-1)[mask]==g['action_targets'][mask]).all()),'OLD_TASK_CORRECTIONS_LOST'
        anchors.append(policy_features(g['actor_features'][mask],native[mask],method[mask]))
    return torch.cat(anchors)


def fit(run):
    u.verify_sources(run);torch.set_num_threads(4);rows,unchanged=gather(run);p=u.read(run/'PROTOCOL.json');results={}
    for arm in ('BC','B2','OURS'):
        fit_rows=[r for r in rows if r['partition']=='FIT' and r['arm']==arm]
        dev_rows=[r for r in rows if r['partition']=='DEV' and r['arm']==arm]
        counts=Counter(r['target'] for r in fit_rows)
        # Both directions must actually have been observed, not synthesized.
        if not counts[1] or not counts[2]:
            results[arm]=dict(status='MISSING_GAIN_OR_HARM_TRAINING_EXAMPLES',counts=dict(counts));continue
        torch.manual_seed(42);model=InterventionGate(3592);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
        x=torch.stack([policy_features(r['actor'],r['native'],r['method']) for r in fit_rows]);y=torch.tensor([r['target'] for r in fit_rows])
        anchor=task_anchors(p,arm);folder=run/'gates'/arm;folder.mkdir(parents=True,exist_ok=True)
        finished=(folder/'FINAL.pt').exists();attempt=time.time_ns()
        if finished:model.load_state_dict(torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True)['model'])
        with (folder/'STEPS.jsonl').open('a',buffering=1) as log:
            for step in range(0 if finished else 2000):
                logits=model.linear(x);a=model.linear(anchor)
                ce=F.cross_entropy(logits,y)
                # Separate known-action retention, not invented navigation successes.
                keep=F.relu(1.6931471805599454+a[:,2]-a[:,1]).mean()
                loss=ce+.5*keep;optimizer.zero_grad(set_to_none=True);loss.backward()
                assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() for t in model.parameters())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if (step+1)%20==0:
                    progress=dict(attempt=attempt,step=step+1,planned=2000,arm=arm,loss=float(loss.detach()),trajectory_ce=float(ce.detach()),
                        separate_known_action_retention=float(keep.detach()),gradient_norm=float(norm),fit_examples=len(fit_rows),known_action_anchors=len(anchor))
                    log.write(json.dumps(progress)+'\n');u.write(folder/'PROGRESS.json',progress)
        if not finished:
            torch.save(dict(model=model.state_dict(),arm=arm,steps=2000,input_fields=['causal_actor_feature','native_logits','method_logits'],
                decision='At first proposal disagreement, once: accept the method for the remaining episode or stay native for the remaining episode'),folder/'FINAL.tmp')
            (folder/'FINAL.tmp').replace(folder/'FINAL.pt')
        with torch.no_grad():
            accepted=choose_method(model.linear(torch.stack([policy_features(r['actor'],r['native'],r['method']) for r in dev_rows]))).tolist() if dev_rows else []
            task_accept=int(choose_method(model.linear(anchor)).sum())
        decisions=[dict(id=r['id'],house=r['house'],accept=accept,target=r['target'],
            native_success=r['native_success'],ungated_success=r['method_success'],selected_success=r['method_success'] if accept else r['native_success']) for r,accept in zip(dev_rows,accepted)]
        fixed=[r for r in unchanged if r['arm']==arm and r['partition']=='DEV'];den=len(decisions)+len(fixed);assert den==40
        native=sum(r['native_success'] for r in decisions+fixed);ungated=sum(r['method_success'] for r in dev_rows+fixed)
        selected=sum(r['selected_success'] for r in decisions)+sum(r['native_success'] for r in fixed)
        results[arm]=dict(status='CPU_GATE_FITTED_AND_LOGGED_BRANCH_DEV_CHECKED',native_success=native,ungated_success=ungated,
            selected_success=selected,planned=den,gains_retained=sum(r['accept'] and r['target']==1 for r in decisions),
            harms_rejected=sum(not r['accept'] and r['target']==2 for r in decisions),decisions=decisions,
            task_action_anchors_accepted=task_accept,task_action_anchors_total=len(anchor),gate_sha256=u.sha(folder/'FINAL.pt'),
            scope='Frozen logged full continuations selected once at their shared causal prefix; not a live gated rollout or unseen SR')
    result=dict(status='CALIBRATION_COMPLETE',results=results,live_gated_policy_evaluated=False,unseen_reused_for_training=False,
        memory_heads_updated=False,gate_steps=2000,claim='Intervention selector prototype; no method efficacy claim')
    u.write(run/'GATE_REVIEW.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();fit(a.run)
