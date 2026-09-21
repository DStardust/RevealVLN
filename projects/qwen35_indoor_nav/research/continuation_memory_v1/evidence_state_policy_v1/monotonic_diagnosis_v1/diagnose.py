"""CPU-only, read-only localization of frozen holdout failures; no policy updates."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
HOLDOUT=BASE/'monotonic_holdout_v1'
sys.path.insert(0,str(HOLDOUT))
from shared import c,read,write,sha,digest,immutable,load,LINE,CPU,V16
from evaluate_continuations import registry,admitted
from evaluator_v16 import legacy,state_sequence,evaluate
from select_action import select,index

def confusion(values):
    counts=Counter();brier=0.
    for probability,truth in values:
        if not math.isfinite(probability) or not 0<=probability<=1 or truth not in (0,1):raise ValueError('INVALID_PROBABILITY_OR_TRUTH')
        counts['tp' if probability>=.5 and truth else 'fp' if probability>=.5 else 'fn' if truth else 'tn']+=1
        brier+=(probability-truth)**2
    return dict(n=len(values),**{k:counts[k] for k in ('tp','fp','fn','tn')},brier=brier/len(values) if values else None,
        sensitivity=counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,
        false_positive_rate=counts['fp']/(counts['fp']+counts['tn']) if counts['fp']+counts['tn'] else None)

def stop_failure(task,truth):
    if task['safe_v16_label']=='PASS':return 'PASS'
    if not task['stopped']:return 'EXHAUSTED' if task['budget_exhausted'] else 'NONSTOP_OTHER'
    if not truth[0]:return 'STOP_WITHOUT_PRIOR_ANCHOR'
    if not truth[2]:return 'STOP_WITHOUT_CURRENT_TERMINAL'
    if task['collisions']:return 'READY_STOP_WITH_COLLISION'
    return 'OTHER'

def first_stationary_forward(trace,cut):
    # Simulator logs retain only episode collision totals. This is a pose diagnostic,
    # not a reconstruction of its hidden per-step collision flag (sliding is possible).
    for t in range(cut,len(trace['actions'])):
        if trace['actions'][t]=='F' and math.dist(trace['observations'][t]['pose']['position'],trace['observations'][t+1]['pose']['position'])<=1e-6:return t
    return None

def evidence_row(row,state,atoms,task):
    z=row['predicted_state'];ready=bool(state[3]);predicted_ready=z[3]>=.5
    return dict(decision=row['decision'],truth=state,predicted_state=z,event_probabilities=row['event_probabilities'],
        action=row['executed_action'],native_action=row['native_action'],method_logits=row['method_logits'],
        stop_margin=row['method_logits'][3]-max(row['method_logits'][:3]),
        ready=ready,predicted_ready=predicted_ready,ready_but_continue=ready and row['executed_action']!='STOP',
        predicted_ready_but_continue=predicted_ready and row['executed_action']!='STOP',
        anchor_event=None if task=='task_T' else bool(atoms['anchor']),terminal_event=bool(atoms['terminal']),
        input_key=row['raw']['input_key'])

def summarize(rows):
    result=dict(n=len(rows),passed=sum(r['status']=='PASS' for r in rows),colliding=sum(r['collisions']>0 for r in rows),
        collision_events=sum(r['collisions'] for r in rows),exhausted=sum(r['exhausted'] for r in rows),
        causes=dict(Counter(r['failure_type'] for r in rows)),actions=dict(sum((Counter(r['actions']) for r in rows),Counter())),
        takeover_actions=dict(Counter(r['takeover']['action'] for r in rows)),
        ever_ready=sum(r['ever_ready'] for r in rows),ever_ready_predicted_and_continue=sum(r['ever_ready_predicted_and_continue'] for r in rows),
        stationary_forward_episodes=sum(r['first_stationary_forward'] is not None for r in rows),
        ever_unseen_history_false_positive=sum(r['ever_unseen_history_false_positive'] for r in rows),
        new_unseen_history_crossings=sum(r['new_unseen_history_crossing'] is not None for r in rows),
        collisions_after_ready_takeover=sum(r['collisions']>0 and r['takeover']['ready'] for r in rows),
        collided_and_exhausted=sum(r['collisions']>0 and r['exhausted'] for r in rows))
    for bit,name in enumerate(('past_anchor','seen_after','current_terminal','ready')):
        result['takeover_'+name]=confusion([(r['takeover']['predicted_state'][bit],r['takeover']['truth'][bit]) for r in rows])
    ready=[r for r in rows if r['takeover']['ready']]
    result['ready_takeover_n']=len(ready)
    result['ready_takeover_stop']=sum(r['takeover']['action']=='STOP' for r in ready)
    result['ready_takeover_predicted_continue']=sum(r['takeover']['predicted_ready_but_continue'] for r in ready)
    return result

def training_support(data):
    rows=[]
    for split in ('FIT','DEV'):
        families=[f for f in data['families'] if f['split']==split];actions=Counter();targets=Counter();lengths=[];supervised_inputs=set();sources=set();events=Counter()
        for f in families:
            for r in f['sequences']:
                lengths.append(len(r['features']));sources.add(r['source_trace']['path'])
                for t,(a,m) in enumerate(zip(r['targets'],r['action_masks'])):
                    targets['FLRS'[a]]+=1
                    if m:
                        actions['FLRS'[a]]+=1;supervised_inputs.add((r['features'][t],a))
                for y,m in zip(r['event_targets'],r['event_masks']):
                    for k,name in enumerate(('anchor','terminal')):
                        if m[k]:events[name+'_'+str(y[k])]+=1
        rows.append(dict(split=split,families=len(families),houses=len({f['house'] for f in families}),sequence_rows=len(lengths),
            unique_physical_traces=len(sources),action_supervised_rows=dict(actions),all_trace_actions_including_masked=dict(targets),
            unique_supervised_feature_action_pairs=len(supervised_inputs),max_sequence=max(lengths),min_sequence=min(lengths),event_supervision=dict(events)))
    ordinary=read(BASE.parent/'natural_transfer_v9/DATA.json')
    distribution=Counter(a for r in ordinary['records'] if r['partition']=='fit' for a in r['targets'])
    return dict(controlled=rows,ordinary_fit_action_rows={str(k):v for k,v in distribution.items()},
        note='Counts from source data, not multiplied optimizer exposure. Ordinary CE remains active; controlled trace actions are not the whole training distribution.')

def main(run,out):
    began=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    cfg=read(run/'PROTOCOL.json');lock=read(run/'SOURCE_LOCK.json')
    for path in (BASE/'model.py',BASE/'objective.py',V16/'evaluator_v16.py',LINE/'data_pipeline/mechanism_factory_v2/compiler.py',HOLDOUT/'evaluate_continuations.py'):
        if sha(path)!=lock['files'][str(path.relative_to(LINE))]:raise ValueError('FROZEN_SOURCE_CHANGED')
    reg=registry(run);groups=admitted(run,reg);data=read(run/'DATA.json');families={f['family_id']:f for f in data['raw_families']}
    saved=read(run/'ROLLOUTS.json');by_rank={r['rank']:r for r in saved}
    manifest={str(p.relative_to(LINE)):sha(p) for p in (run/'RESULT.json',run/'ROLLOUTS.json',run/'DATA.json',run/'EVALUATION_REGISTRY.json',run/'SOURCE_LOCK.json',CPU/'DATA.json')}
    episodes=[];buckets=defaultdict(list);pairs={};all_steps=0;monotonic_errors=[]
    for slot in reg['slots']:
        condition=reg['conditions'][slot['condition']]
        if slot['condition'] not in groups:continue
        session=groups[slot['condition']];folder=session/'rollouts'/f"{slot['rank']:04d}";family=families[condition['family_id']]
        steps=c.records(folder/'POLICY_STEPS.jsonl');trace=read(folder/'TRACE_PRIVILEGED.json');task=read(folder/'TASK_RESULT.json')
        compiler=legacy.Compiler(**family['compiler']);states=state_sequence(compiler,trace['observations'],condition['task_id']);atoms=compiler.atoms(trace['observations'])
        cut=len(family['histories'][condition['history_id']]);recomputed=evaluate(compiler,trace,condition['task_id'],cut)
        if any(task[k]!=v for k,v in recomputed.items()) or task['safe_v16_label']!=by_rank[slot['rank']]['status']:raise ValueError('LABEL_CHANGED')
        if len(steps)!=len(trace['actions'])-cut or [s['decision'] for s in steps]!=list(range(cut,len(trace['actions']))):raise ValueError('DECISION_ALIGNMENT')
        decision_rows=[];first_false=None;crossing=None;prior=None;repeat=Counter();ever_ready_pred=False;no_ready=Counter();recurrence_error=0.
        for s in steps:
            t=s['decision'];z=s['predicted_state'];truth=states[t];executed=s['executed_action']
            if select(s['logits'],s['method_logits'])['executed_action']!=executed or c.ACTIONS['FLRS'.index(trace['actions'][t])]!=executed:raise ValueError('EXECUTED_ACTION_CHANGED')
            entry=evidence_row(s,truth,atoms[t],condition['task_id']);decision_rows.append(entry);repeat[s['raw']['input_key']]+=1
            if truth[3] and z[3]>=.5 and executed!='STOP':ever_ready_pred=True
            if not truth[1] and z[1]>=.5:
                if first_false is None:first_false=t
                if prior is not None and prior['predicted_state'][1]<.5 and crossing is None:crossing=t
            if slot['arm']=='MONOTONIC':
                error=abs(z[1]-(z[0]+(1-z[0])*s['event_probabilities'][0]))
                if prior is not None:error=max(error,abs(z[0]-prior['predicted_state'][1]))
                recurrence_error=max(recurrence_error,error)
                if prior is not None and z[1]<prior['predicted_state'][1]-1e-6:monotonic_errors.append([slot['rank'],t])
            for bit,name in enumerate(('past_anchor','seen_after','current_terminal','ready')):
                # Autonomous step counts are length-biased; takeover stats below weight episodes equally.
                for seed in (None,slot['seed']):buckets[slot['arm'],seed,condition['endpoint'],name].append((z[bit],truth[bit]))
            if not truth[1]:no_ready['unseen_decisions']+=1;no_ready['unseen_prediction_sum']+=z[1]
            prior=entry
        episode=dict(slot,**condition,status=task['safe_v16_label'],path=str(folder.relative_to(LINE)),
            collisions=task['collisions'],exhausted=task['budget_exhausted'],decisions=task['total_decisions'],autonomous_decisions=len(steps),
            failure_type=stop_failure(task,states[steps[-1]['decision']]),takeover=decision_rows[0],last=decision_rows[-1],
            ever_ready=any(r['ready'] for r in decision_rows),ever_ready_predicted_and_continue=ever_ready_pred,
            ready_decisions=sum(r['ready'] for r in decision_rows),ready_continue_decisions=sum(r['ready_but_continue'] for r in decision_rows),
            ever_unseen_history_false_positive=first_false is not None,first_unseen_history_false_positive=first_false,new_unseen_history_crossing=crossing,
            first_stationary_forward=first_stationary_forward(trace,cut),actions=dict(Counter(s['executed_action'] for s in steps)),
            repeated_input_decisions=sum(n-1 for n in repeat.values()),max_monotonic_recurrence_error=recurrence_error,
            first_ready_continue=next((r for r in decision_rows if r['ready_but_continue']),None),
            first_forward=next((r for r in decision_rows if r['action']=='move_forward'),None),
            prefix_steps=cut,anchor_age_at_takeover=family['certificate']['anchor_age'][condition['history_id']])
        episodes.append(episode);pairs[slot['condition'],slot['seed'],slot['arm']]=(episode,decision_rows);all_steps+=len(steps)
        if len(episodes)%100==0:print('episodes',len(episodes),'decisions',all_steps,flush=True)
    if len(episodes)!=read(run/'RESULT.json')['complete'] or monotonic_errors:raise ValueError('DENOMINATOR_OR_MONOTONICITY')
    contrasts=[]
    for condition,seed,arm in sorted(pairs):
        if arm!='MONOTONIC':continue
        b,bs=pairs[condition,seed,arm];a,ass=pairs[condition,seed,'DIRECT']
        first_difference=next((dict(direct=x,monotonic=y) for x,y in zip(ass,bs) if x['action']!=y['action']),None)
        contrasts.append(dict(condition=condition,seed=seed,endpoint=b['endpoint'],house=b['house'],stratum=b['stratum'],history=b['history_id'],
            direct_rank=a['rank'],monotonic_rank=b['rank'],direct_status=a['status'],monotonic_status=b['status'],
            outcome='tie' if a['status']==b['status'] else 'win' if b['status']=='PASS' else 'loss',
            extra_collision=b['collisions']>0 and a['collisions']==0,avoided_collision=a['collisions']>0 and b['collisions']==0,
            direct_collision_events=a['collisions'],monotonic_collision_events=b['collisions'],
            direct_failure=a['failure_type'],monotonic_failure=b['failure_type'],
            direct_takeover=a['takeover'],monotonic_takeover=b['takeover'],first_action_divergence=first_difference))
    aggregates=[]
    for endpoint in ('main','control'):
        for arm in cfg['arms']:
            for seed in [None]+cfg['seeds']:
                selected=[r for r in episodes if r['endpoint']==endpoint and r['arm']==arm and (seed is None or r['seed']==seed)]
                aggregates.append(dict(endpoint=endpoint,arm=arm,seed=seed,**summarize(selected)))
    support=training_support(read(CPU/'DATA.json'))
    weights={}
    import torch
    for row in read(run/'MODEL_REUSE.json')['models']:
        p=run/'train'/row['model']/'FINAL.pt'
        if sha(p)!=row['sha256']:raise ValueError('HEAD_CHANGED')
        state=torch.load(p,map_location='cpu',weights_only=True)
        weights[row['model']]=state['state_action.weight'].tolist();manifest[str(p.relative_to(LINE))]=row['sha256']
    # Pure readout algebra on logged states, no recurrent intervention and no new trajectory.
    readout=[]
    for e in episodes:
        for phase in ('takeover','first_ready_continue','first_forward'):
            r=e[phase]
            if r is None:continue
            w=weights[e['model']];z=r['predicted_state'];g=r['truth']
            contribution=[sum(a*b for a,b in zip(wi,z)) for wi in w]
            core=[v-d for v,d in zip(r['method_logits'],contribution)]
            exact=[v+sum(a*b for a,b in zip(wi,g)) for v,wi in zip(core,w)]
            readout.append(dict(rank=e['rank'],model=e['model'],endpoint=e['endpoint'],phase=phase,truth_ready=bool(g[3]),
                actual_action=r['action'],without_state_action=c.ACTIONS[index(core)],exact_state_algebra_action=c.ACTIONS[index(exact)],
                state_contribution=contribution,observed_stop_margin=r['stop_margin'],algebra_stop_margin=exact[3]-max(exact[:3])))
    immutable(out/'EPISODES.json',episodes);immutable(out/'PAIRED_FAILURES.json',contrasts);immutable(out/'READOUT_ALGEBRA.json',readout)
    immutable(out/'TRAINING_SUPPORT.json',support)
    immutable(out/'RESULT.json',dict(status='READ_ONLY_DIAGNOSIS_COMPLETE',episodes=len(episodes),planned=len(reg['slots']),
        missing=len(reg['slots'])-len(episodes),groups=len(groups),autonomous_decisions=all_steps,aggregates=aggregates,
        calibration=[dict(arm=a,seed=s,endpoint=e,target=t,**confusion(v)) for (a,s,e,t),v in buckets.items()],
        gpu_hours=0,new_optimizer_updates=0,new_environment_actions=0,seconds=time.monotonic()-began,
        limitations=['Post-hoc diagnosis on now exposed holdout, no new efficacy experiment.',
            '0.5 is a fixed descriptive threshold for state errors; not a deployed action threshold or tuned hyperparameter.',
            'Collision flag time was not logged; zero-displacement forward is a separate kinematic proxy.',
            'Readout algebra with exact state is oracle offline localization, not deployable input or counterfactual SR.',
            'No pre-takeover event probabilities in original logs: cannot separate initial prior error from old event false positives.',
            'All post-divergence comparisons concern different visited states; takeover comparisons share actual inputs.',
            'Final held-out method weights were not changed; only six small action matrices were loaded on CPU.']))
    for condition,session in groups.items():
        for name in (f'GROUP_{condition:03d}.json',f'STATE_SEAL_{condition:03d}.json'):
            p=session/name;manifest[str(p.relative_to(LINE))]=sha(p)
    immutable(out/'INPUT_MANIFEST.json',dict(files=manifest,group_manifest_scope='admitted() checked SHA of every consumed rollout file against these sealed group manifests',
        diagnostic_source_sha256=sha(Path(__file__)),source_run=str(run)))
    print('COMPLETE',len(episodes),all_steps,'GPU=0',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HOLDOUT/'runs/holdout_001');p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.run,args.output)
