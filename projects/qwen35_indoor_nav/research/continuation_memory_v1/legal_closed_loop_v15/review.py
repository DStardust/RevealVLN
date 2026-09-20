"""Recompute full denominators and distinguish teacher diagnostics from navigation."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import statistics
import sys
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def percentile(values,q):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*q;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def training():
    protocol=c.read(HERE/'TRAIN_PROTOCOL.json');run=HERE/'train_run_001';rows=[]
    for size,fit_ids in protocol['data_sizes'].items():
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                tag=f'{size}_{arm}_{seed}';path=run/(tag+'_RESULT.json')
                if not path.exists():continue
                result=c.read(path);assert result['checkpoint_readback_verified']
                assert len(c.records(run/(tag+'_STEPS.jsonl')))==result['updates']==600
                scope=defaultdict(list)
                for family in result['families']:
                    group='check' if family['split']=='check' else 'trained_fit' if family['family_id'] in fit_ids else 'untrained_fit'
                    scope[group].append(family)
                sums={}
                for group,families in scope.items():
                    cases=[case for f in families for case in f['fork_cases']]
                    sums[group]=dict(families=len(families),houses=len({f['house'] for f in families}),
                        fork_pairs=len(cases),both_correct={mode:sum(x['both_correct'][mode] for x in cases) for mode in ('correct','wrong','sham')},
                        action_correct=sum(f['action_correct'] for f in families),action_owners=sum(f['action_owners'] for f in families),
                        mean_action_ce=statistics.mean(f['action_ce'] for f in families),
                        query_correct=sum(f['query_correct'] for f in families) if arm!='B1' else None,
                        query_cells=sum(f['query_cells'] for f in families))
                natural=result['ordinary_teacher_check']
                rows.append(dict(tag=tag,size=size,seed=seed,arm=arm,updates=result['updates'],scopes=sums,
                    ordinary_teacher=dict(routes=natural['routes'],correct=natural['correct'],decisions=natural['decisions'],
                        base_correct=sum(x['base_correct'] for x in natural['rows']),
                        base_mean_route_ce=statistics.mean(x['base_ce'] for x in natural['rows']),
                        ce=natural['mean_route_ce'],premature_stops=sum(x['premature_stops'] for x in natural['rows'])),
                    initial_state_sha256=result['initial_state_sha256'],seconds=result['seconds']))
    for size in protocol['data_sizes']:
        for seed in protocol['seeds']:
            hashes={row['initial_state_sha256'] for row in rows if row['size']==size and row['seed']==seed}
            assert len(hashes)<=1,'UNMATCHED_INITIALIZATION'
    return dict(completed_models=len(rows),expected_models=18,rows=rows,
        actual_updates=sum(x['updates'] for x in rows),base_updates=0,
        metric_scope='Teacher-conditioned action/fork diagnostics, without native STOP guard; not autonomous task success or SR.',
        uncertainty='Three seeds reuse the same three CHECK families in one house. Cases and seeds are correlated; no independent-house confidence claim.')


def admitted_conditions(runs,protocol):
    """A resumed result needs the whole fixed group and a matching final state seal."""
    admitted={};identities={};audits=[]
    for run in runs:
        if not (run/'STATE_SEAL.json').exists():continue
        seal=c.read(run/'STATE_SEAL.json')
        if not (seal['base_unchanged'] and seal['heads_unchanged']):continue
        for path in sorted(run.glob('CONDITION_*_AUDITS.json')):
            condition=int(path.name.split('_')[1])
            expected={rank for rank,row in enumerate(protocol['rollouts']) if row['condition']==condition}
            assert expected<=set(seal['completed_ranks']),'UNSEALED_CONDITION'
            assert condition not in admitted,'DUPLICATE_COMPLETE_CONDITION'
            identity=c.read(run/'METHOD_IDENTITY.json')
            assert seal['base_state_sha256']==identity['base_state_sha256'],'BASE_SEAL_MISMATCH'
            assert seal['head_states']==identity['head_states'],'HEAD_SEAL_MISMATCH'
            if identities:
                reference=next(iter(identities.values()))
                assert identity['base_state_sha256']==reference['base_state_sha256'] and identity['head_states']==reference['head_states']
            identities[run.name]=identity;admitted[condition]=run;audits.extend(c.read(path))
    return admitted,audits


def continuations(runs):
    protocol=c.read(HERE/'CONTINUATION_PROTOCOL.json')
    compiler_module=c.load('v15_review_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    raw_module=c.load('v15_review_raw',LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/common.py')
    families={f['family_id']:f for f in c.read(LINE/protocol['raw_config'])['families']}
    admitted,audits=admitted_conditions(runs,protocol)
    sealed=set()
    rows=[];decision_logs={};terminal_observations={};cache_rows=[];prefill_identities={}
    for rank,assignment in enumerate(protocol['rollouts']):
        run=admitted.get(assignment['condition'])
        if run is None:continue
        folder=run/'rollouts'/f'{rank:03d}';record=folder/'ROLLOUT.json'
        assert record.exists(),'MISSING_MEMBER_OF_COMPLETE_CONDITION'
        result=c.read(record);trace=c.read(folder/'TRACE_PRIVILEGED.json');steps=c.records(folder/'POLICY_STEPS.jsonl')
        assert result['rank']==rank and result['model']==assignment['model']
        sealed.add(rank)
        assert result['policy_log_sha256']==c.sha(folder/'POLICY_STEPS.jsonl')
        assert result['trace_sha256']==c.sha(folder/'TRACE_PRIVILEGED.json')
        condition=protocol['conditions'][assignment['condition']];family=families[condition['family_id']]
        history=family['candidate']['histories'][condition['history_id']];cut=len(history)
        original=c.read(LINE/protocol['raw_run']/family['family_id']/(condition['history_id']+'__C0.json'))
        assert trace['actions'][:cut]==history
        assert len(trace['actions'])==cut+len(steps)<=500
        assert trace['actions'][-1]=='S' or len(trace['actions'])==500
        assert all(a['rgb_hash']==b['rgb_hash'] and a['semantic_hash']==b['semantic_hash'] and
            raw_module.exact_pose_equal(a['pose'],b['pose']) for a,b in zip(trace['observations'][:cut+1],original['observations'][:cut+1]))
        compiler=compiler_module.Compiler(**family['compiler']);instruction=compiler.tasks[condition['task_id']]['instruction']
        for t,row in enumerate(steps,cut):
            assert row['decision']==t and row['raw']['instruction']==instruction
            assert row['raw']['rgb_sha256']==[o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]]
            assert row['raw']['executed_history']==[c.ACTIONS['FLR'.index(a)] for a in trace['actions'][max(0,t-8):t]]
            assert row['executed_action']==c.ACTIONS['FLRS'.index(trace['actions'][t])]
            assert all(math.isfinite(v) for v in row['logits']+row['method_logits'])
            native=c.ACTIONS[max(range(4),key=row['logits'].__getitem__)]
            chosen=native if native=='STOP' else c.ACTIONS[max(range(4),key=row['method_logits'].__getitem__)]
            assert native==row['native_action'] and chosen==row['executed_action']
        label=compiler.evaluate(trace,condition['task_id']);assert label==result['task_result']['label']
        events=compiler.atoms(trace['observations']);task=compiler.tasks[condition['task_id']]
        anchor_seen=False;ready=[]
        for event in events:
            ready.append(anchor_seen and bool(event[task['terminal']]))
            anchor_seen=anchor_seen or bool(event[task['anchor']])
        stopped=trace['actions'][-1]=='S'
        terminal_valid=ready[-1] if stopped else False
        decision_logs[(assignment['condition'],assignment['model'])]=steps
        terminal_observations[(assignment['condition'],assignment['model'])]=trace['observations'][-1]
        prefill=c.read(folder/'PREFILL_AUDIT.json')
        assert prefill['prefix_steps']==cut and prefill['all_raw_inputs_matched']
        assert prefill['live_feature_origin_rank']==rank-rank%18
        if assignment['condition'] in prefill_identities:
            assert prefill['features']==prefill_identities[assignment['condition']],'PREFILL_FEATURE_MISMATCH'
        prefill_identities[assignment['condition']]=prefill['features']
        measured_prefix=c.records(folder/'PREFILL.jsonl') if rank%18==0 else []
        for item in steps+measured_prefix:
            if 'training_cache_comparison' in item:cache_rows.append(item['training_cache_comparison'])
        counts=Counter(row['raw']['input_key'] for row in steps)
        rows.append(dict(rank=rank,condition=assignment['condition'],model=assignment['model'],session=run.name,sealed=True,
            label=label,collisions=trace['collisions'],total_decisions=len(trace['actions']),autonomous_decisions=len(steps),
            repeated_decisions=sum(n-1 for n in counts.values()),stopped=trace['actions'][-1]=='S',
            native_stop_guard_uses=sum(s['native_stop_guard_applied'] for s in steps),
            reached_ready=any(ready),ready_at_stop=terminal_valid,
            stopped_without_terminal=stopped and not bool(events[-1][task['terminal']]),
            stopped_without_prior_anchor=stopped and not any(bool(e[task['anchor']]) for e in events[:-1]),
            reached_ready_but_no_valid_stop=any(ready) and not terminal_valid,
            budget_exhausted=not stopped,stop_native_margin=(steps[-1]['logits'][3]-max(steps[-1]['logits'][:3])) if stopped else None,
            inference_seconds=[s['inference_seconds'] for s in steps],controller_seconds=[s['controller_seconds'] for s in steps]))
    # Recompute the retained audit records; never trust a saved boolean alone.
    expected_audits={(condition,f'{size}_{arm}_{seed}',f'{size}_Ours_{seed}')
        for condition in admitted for size in ('S1','L3') for seed in (1209,1210,1211) for arm in ('B1','B2')}
    assert len(audits)==len(expected_audits) and {(a['condition'],a['left'],a['right']) for a in audits}==expected_audits,'INCOMPLETE_PREFIX_AUDITS'
    for audit in audits:
        left=decision_logs[(audit['condition'],audit['left'])];right=decision_logs[(audit['condition'],audit['right'])]
        checks=[];first=None
        for a,b in zip(left,right):
            comparison=c.prefix_compare(a,b);checks.append(comparison)
            assert comparison['input_prefix_matched'] and not comparison['argmax_flip_count'],'INVALID_NATIVE_PREFIX'
            if a['executed_action']!=b['executed_action']:
                first=a['decision'];break
        if first is None:
            assert len(left)==len(right),'UNMATCHED_TERMINATION'
            a=terminal_observations[(audit['condition'],audit['left'])];b=terminal_observations[(audit['condition'],audit['right'])]
            assert a['rgb_hash']==b['rgb_hash'] and a['semantic_hash']==b['semantic_hash'] and raw_module.exact_pose_equal(a['pose'],b['pose']),'UNMATCHED_TERMINAL_STATE'
        assert audit['first_method_action_difference']==first and audit['decisions']==len(checks)
        assert audit['logits_bitwise_equal']==all(x['logits_bitwise_equal'] for x in checks)
        assert audit['max_logit_delta']==max(x['max_logit_delta'] for x in checks)
        assert audit['input_prefix_matched'] and audit['action_prefix_matched'] and audit['argmax_flip_count']==0
    by_model={}
    for tag in protocol['models']:
        selected=[row for row in rows if row['model']==tag and row['sealed']]
        labels=Counter(row['label'] for row in selected)
        inference=[v for row in selected for v in row['inference_seconds']];control=[v for row in selected for v in row['controller_seconds']]
        by_model[tag]=dict(complete=len(selected),expected=12,pass_count=labels['pass'],fail_count=labels['fail'],unknown_count=labels['unknown'],
            full_denominator_pass_lower_bound=labels['pass']/12 if len(selected)==12 else None,
            collision_episodes=sum(row['collisions']>0 for row in selected),collisions=sum(row['collisions'] for row in selected),
            total_actions=sum(row['total_decisions'] for row in selected),autonomous_actions=sum(row['autonomous_decisions'] for row in selected),
            repeated_decisions=sum(row['repeated_decisions'] for row in selected),stops=sum(row['stopped'] for row in selected),
            failure_descriptors={name:sum(row[name] for row in selected) for name in ('budget_exhausted','reached_ready','ready_at_stop',
                'stopped_without_terminal','stopped_without_prior_anchor','reached_ready_but_no_valid_stop')},
            p50_inference_seconds=percentile(inference,.5),p95_inference_seconds=percentile(inference,.95),
            p50_controller_seconds=percentile(control,.5),p95_controller_seconds=percentile(control,.95))
    comparisons=[]
    valid={(a['condition'],a['left'],a['right']) for a in audits if a['input_prefix_matched'] and a['action_prefix_matched']}
    lookup={(row['condition'],row['model']):row for row in rows if row['sealed']}
    for size in ('S1','L3'):
        for seed in (1209,1210,1211):
            right=f'{size}_Ours_{seed}'
            for arm in ('B1','B2'):
                left=f'{size}_{arm}_{seed}';wins=[];losses=[];unknown=[];missing=[]
                for condition in range(12):
                    a,b=lookup.get((condition,left)),lookup.get((condition,right))
                    if not a or not b or (condition,left,right) not in valid:missing.append(condition);continue
                    if 'unknown' in (a['label'],b['label']):unknown.append(condition);continue
                    if a['label']=='fail' and b['label']=='pass':wins.append(condition)
                    if a['label']=='pass' and b['label']=='fail':losses.append(condition)
                comparisons.append(dict(left=left,right=right,wins=wins,losses=losses,unknown=unknown,missing_or_unmatched=missing,
                    paired_delta_pass=(len(wins)-len(losses))/12 if not unknown and not missing else None))
    return dict(completed_rollouts=len(rows),sealed_rollouts=len(sealed),expected_rollouts=216,
        complete_conditions=sorted(admitted),sessions={str(k):v.name for k,v in admitted.items()},
        session_runtime_identity={run.name:c.read(run/'RUNTIME_IDENTITY.json') for run in set(admitted.values())},
        metrics_recomputed=True,models=by_model,comparisons=comparisons,
        prefix_comparisons=len(audits),logits_bitwise_equal=all(a['logits_bitwise_equal'] for a in audits) if audits else None,
        max_logit_delta=max((a['max_logit_delta'] for a in audits),default=None),
        argmax_flip_count=sum(a['argmax_flip_count'] for a in audits),
        cache_live_diagnostic=dict(comparisons=len(cache_rows),processed_inputs_all_equal=all(x['processed_inputs_equal'] for x in cache_rows),
            max_feature_delta=max((x['max_feature_delta'] for x in cache_rows),default=None),
            max_relative_feature_l2=max((x['relative_feature_l2'] for x in cache_rows),default=None),
            max_logit_delta=max((x['max_logit_delta'] for x in cache_rows),default=None),
            native_argmax_flips=sum(x['native_argmax_flip'] for x in cache_rows),
            scope='Across extraction/live sessions on recurring identical processed inputs; distinct from within-group causal prefix checks.'),
        records=[{k:v for k,v in row.items() if k not in ('inference_seconds','controller_seconds')} for row in rows],
        scientific_scope='Fixed-history autonomous SEE2 continuation in one new CHECK house. UNKNOWN kept distinct. Not R2R SR, blind-test generalization, or deployment.')


def main():
    trained=training();c.write(HERE/'TRAINING_REVIEW.json',trained)
    runs=sorted(HERE.glob('continuation_run_*'))
    if any((run/'STATE_SEAL.json').exists() for run in runs):
        measured=continuations(runs);c.write(HERE/'CONTINUATION_REVIEW.json',measured)
    print(dict(complete_models=trained['completed_models'],updates=trained['actual_updates']))


if __name__=='__main__':main()
