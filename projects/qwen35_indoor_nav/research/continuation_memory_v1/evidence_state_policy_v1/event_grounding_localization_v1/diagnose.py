"""CPU-only localization of every registered main pair and its first action divergence."""
import argparse
from collections import Counter
import itertools
from pathlib import Path
import sys
import time
HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent/'event_grounding_repair_v1'
sys.path.insert(0, str(RUNTIME))
from shared import c, read, sha, immutable, write, digest, LINE, make_head
from evaluate_continuations import registry, admitted, prefix_audit
from evaluator_v16 import legacy, state_sequence, evaluate
from select_action import select

ARMS = ('ORIGINAL', 'EVENT')


def input_signature(raw):
    # Runtime input_key and cache key use different hash namespaces. Match the
    # actual causal fields, as the frozen runtime's cache/live audit does.
    return (raw['instruction'], tuple(raw['rgb_sha256']), tuple(raw['executed_history']))


def action(values):
    import torch
    if not torch.isfinite(values).all() or values.shape != (4,):
        raise ValueError('INVALID_ACTION_VECTOR')
    return c.ACTIONS[int(values.argmax())]


def decomposition(left, right, weights):
    """Algebraic single-input swaps; no claim about counterfactual trajectories."""
    import torch
    rows = dict(zip(ARMS, (left, right)))
    states = {a: torch.tensor(rows[a]['predicted_state'], dtype=torch.float64) for a in ARMS}
    logits = {a: torch.tensor(rows[a]['method_logits'], dtype=torch.float64) for a in ARMS}
    contribution = {a: weights[a] @ states[a] for a in ARMS}
    core = {a: logits[a]-contribution[a] for a in ARMS}
    swaps = {}
    for component, z, w in itertools.product(ARMS, repeat=3):
        values = core[component]+weights[w] @ states[z]
        swaps['/'.join((component, z, w))] = dict(action=action(values), logits=values.tolist())
    oi = c.ACTIONS.index(left['executed_action']); ei = c.ACTIONS.index(right['executed_action'])
    def margin(v):
        return float(v[oi]-v[ei])
    core_delta = margin(core['EVENT']-core['ORIGINAL'])
    state_delta = margin(contribution['EVENT']-contribution['ORIGINAL'])
    total_delta = margin(logits['EVENT']-logits['ORIGINAL'])
    if abs(core_delta+state_delta-total_delta) > 1e-8:
        raise ValueError('SCORE_DECOMPOSITION_FAILED')
    return dict(core_logits={a: v.tolist() for a, v in core.items()},
        state_branch={a: v.tolist() for a, v in contribution.items()},
        action_margin_original_vs_event={a: margin(logits[a]) for a in ARMS},
        margin_delta=dict(core=core_delta, state_branch=state_delta, total=total_delta),
        dominant_adverse_component='core_action_path' if core_delta < state_delta else 'state_branch',
        swaps=swaps, swap_key_order=['core_action_path', 'predicted_state', 'state_action_weights'],
        state_only_restores_original=swaps['EVENT/ORIGINAL/EVENT']['action']==left['executed_action'],
        state_branch_only_restores_original=swaps['EVENT/ORIGINAL/ORIGINAL']['action']==left['executed_action'],
        core_only_restores_original=swaps['ORIGINAL/EVENT/EVENT']['action']==left['executed_action'])


def replay(rows, models, cache, lookup, processed_inputs, native, logged):
    """Replay the complete common causal prefix only if every feature is present."""
    import torch
    missing = [t for t, row in enumerate(rows) if input_signature(row['raw']) not in lookup]
    if missing:
        return dict(status='UNAVAILABLE_CAUSAL_FEATURES', missing_steps=missing,
                    reason='No new base-model forward or environment execution authorized for this diagnostic.')
    ids = [lookup[input_signature(row['raw'])] for row in rows]
    if any(row['processed'] != processed_inputs[i] for row, i in zip(rows, ids)):
        raise ValueError('CACHED_PROCESSED_INPUT_MISMATCH')
    outputs = {}; memories = {}
    with torch.inference_mode():
        for arm in ARMS:
            net = models[arm]; state = net.reset(1, 'cpu')
            for i in ids:
                values, state, detail = net.step(cache['features'][i:i+1], torch.tensor([native]), state)
            outputs[arm] = values[0]; memories[arm] = state.memory
        errors = {arm: float((outputs[arm]-torch.tensor(logged[arm]['method_logits'])).abs().max()) for arm in ARMS}
        matched = all(action(outputs[a])==logged[a]['executed_action'] and errors[a] <= 1e-4 for a in ARMS)
        if not matched:
            return dict(status='CACHE_CPU_LIVE_PARITY_NOT_MET', max_logit_delta=errors,
                        actions={a: action(outputs[a]) for a in ARMS}, tolerance=1e-4)
        feature = cache['features'][ids[-1]:ids[-1]+1]
        base_logits = torch.tensor([native])
        # Hold EVENT's current state branch fixed, independently swap the recurrent
        # memory tensor and core action readout. These are local diagnostic inputs.
        state_branch = models['EVENT'].state_action(torch.tensor([logged['EVENT']['predicted_state']]))
        swaps = {}
        for memory_arm, actor_arm in itertools.product(ARMS, repeat=2):
            values = models[actor_arm].core.action_logits(memories[memory_arm], base_logits, feature)+state_branch
            swaps[f'{memory_arm}/{actor_arm}'] = dict(action=action(values[0]), logits=values[0].tolist())
    return dict(status='CPU_COMMON_PREFIX_REPRODUCED', causal_steps=len(ids), max_logit_delta=errors,
        swap_key_order=['recurrent_memory_tensor', 'core_action_readout_weights'], swaps=swaps,
        actor_only_restores_original=swaps['EVENT/ORIGINAL']['action']==logged['ORIGINAL']['executed_action'],
        memory_only_restores_original=swaps['ORIGINAL/EVENT']['action']==logged['ORIGINAL']['executed_action'],
        note='Local action interventions only, not simulated recovery, SR, or proof a hybrid checkpoint is usable.')


def summarize(cases):
    summaries = {}
    for category in ('loss', 'win', 'both_pass', 'both_fail'):
        selected = [r for r in cases if r['outcome'] == category]
        diffs = [r for r in selected if r['first'] is not None]
        def count(fn):
            return sum(bool(fn(r)) for r in diffs)
        reproduced = [r for r in diffs if r.get('cpu_replay', {}).get('status') == 'CPU_COMMON_PREFIX_REPRODUCED']
        summaries[category] = dict(n=len(selected), first_action_divergences=len(diffs),
            first_at_takeover=count(lambda r: r['first']['offset']==0),
            initial_state_classes_same=count(lambda r: r['first']['state_classes_same']),
            history_classes_same=count(lambda r: r['first']['history_classes_same']),
            original_stop_event_continue=count(lambda r: r['first']['actions']['ORIGINAL']=='STOP' and r['first']['actions']['EVENT']!='STOP'),
            both_recognize_ready=count(lambda r: r['first']['both_recognize_ready']),
            ready_truth=count(lambda r: r['first']['truth'][3]),
            ready_recognized_event_continues=count(lambda r: r['first']['truth'][3] and r['first']['both_recognize_ready'] and r['first']['actions']['EVENT']!='STOP'),
            dominant_components=dict(Counter(r['decomposition']['dominant_adverse_component'] for r in diffs)),
            state_only_restores_original=count(lambda r: r['decomposition']['state_only_restores_original']),
            state_branch_only_restores_original=count(lambda r: r['decomposition']['state_branch_only_restores_original']),
            core_only_restores_original=count(lambda r: r['decomposition']['core_only_restores_original']),
            history_types=dict(Counter(r['condition']['history_id'] for r in selected)),
            strata=dict(Counter(r['condition']['stratum'] for r in selected)),
            parent_families=len({r['condition']['parent_family_id'] for r in selected}),
            distinct_first_raw_inputs=len({r['first']['input_key'] for r in diffs}),
            event_final=dict(Counter(r['final']['EVENT']['failure_type'] for r in selected)),
            cpu_replay_status=dict(Counter(r.get('cpu_replay', {}).get('status', 'NOT_REQUESTED') for r in diffs)),
            actor_only_restores_original=sum(r['cpu_replay']['actor_only_restores_original'] for r in reproduced),
            memory_only_restores_original=sum(r['cpu_replay']['memory_only_restores_original'] for r in reproduced))
    return summaries


def main(run, out):
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False); began=time.monotonic()
    reg=registry(run); groups=admitted(run, reg)
    if len(groups)!=128 or read(run/'RESULT.json')['complete']!=768:
        raise ValueError('INCOMPLETE_SOURCE_STUDY')
    data=read(run/'DATA.json'); raw={f['family_id']:f for f in data['raw_families']}
    cache_path=run/'features/FEATURES.pt'
    if sha(cache_path)!=read(run/'features/FEATURE_RESULT.json')['file_sha256']:
        raise ValueError('SOURCE_CACHE_HASH')
    cache={k:v.float() for k,v in torch.load(cache_path,map_location='cpu',weights_only=True).items()}
    lookup={(r['instruction'],tuple(h[7:] for h in r['rgb_refs']),tuple(r['executed'])):i for i,r in enumerate(data['features'])}
    processed_inputs={}
    for path in (run/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        if not (path.parent/'STATE_SEAL.json').exists():continue
        for row in c.records(path):processed_inputs[row['index']]=row['processed']
    nets={}; before={}; manifest={}
    def bind(path):
        manifest[str(path.relative_to(LINE))]=sha(path)
    for path in [Path(__file__),cache_path,run/'RESULT.json',run/'PROTOCOL.json',run/'SOURCE_LOCK.json',run/'DATA.json',run/'EVALUATION_REGISTRY.json']:
        bind(path)
    for relative in ['research/continuation_memory_v1/evidence_state_policy_v1/model.py',
                     'research/continuation_memory_v1/grounded_state_transfer_v16/evaluator_v16.py',
                     'data_pipeline/mechanism_factory_v2/compiler.py']:
        if sha(LINE/relative)!=read(run/'SOURCE_LOCK.json')['files'][relative]:
            raise ValueError('FROZEN_SOURCE_CHANGED:'+relative)
        bind(LINE/relative)
    for seed in (1209,1210,1211):
        for arm in ARMS:
            tag=f'{arm}_{seed}'; folder=run/'train'/tag; result=read(folder/'RESULT.json'); path=folder/'FINAL.pt'
            if sha(path)!=result['checkpoint_sha256']:raise ValueError('HEAD_HASH')
            net=make_head(tag).eval(); net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            for parameter in net.parameters():parameter.requires_grad_(False)
            if c.model_identity(net)['sha256']!=result['final']:raise ValueError('MODEL_IDENTITY')
            nets[tag]=net; before[tag]=result['final']; bind(path)
    cases=[]; native_delta=0.; replayed_steps=0
    for ci,cond in enumerate(reg['conditions']):
        if cond['endpoint']!='main':continue
        session=groups[ci]; family=raw[cond['family_id']]; cutoff=len(family['histories'][cond['history_id']])
        compiler=legacy.Compiler(**family['compiler'])
        bind(session/f'GROUP_{ci:03d}.json'); bind(session/f'STATE_SEAL_{ci:03d}.json')
        for seed in (1209,1210,1211):
            entries={}; final={}; folders={}
            for arm in ARMS:
                slot=next(r for r in reg['slots'] if r['condition']==ci and r['seed']==seed and r['arm']==arm)
                folder=session/'rollouts'/f"{slot['rank']:04d}";folders[arm]=folder
                steps=c.records(folder/'POLICY_STEPS.jsonl'); trace=read(folder/'TRACE_PRIVILEGED.json'); task=read(folder/'TASK_RESULT.json')
                checked=evaluate(compiler,trace,'task_A',cutoff)
                if any(task[k]!=v for k,v in checked.items()):raise ValueError('TASK_RECOMPUTATION')
                states=state_sequence(compiler,trace['observations'],'task_A')
                atoms=compiler.atoms(trace['observations'])
                if [r['decision'] for r in steps]!=list(range(cutoff,len(trace['actions']))):raise ValueError('STEP_ALIGNMENT')
                for row in steps:
                    if select(row['logits'],row['method_logits'])['executed_action']!=row['executed_action']:raise ValueError('ACTION_RECOMPUTATION')
                    if row['executed_action']!=c.ACTIONS['FLRS'.index(trace['actions'][row['decision']])]:raise ValueError('TRACE_ACTION_ALIGNMENT')
                last=steps[-1]; z=states[last['decision']]
                failure=('PASS' if task['safe_v16_label']=='PASS' else 'BUDGET_EXHAUSTED' if task['budget_exhausted'] else
                    'STOP_BEFORE_HISTORY' if task['stopped'] and not z[0] else 'STOP_WITHOUT_TERMINAL' if task['stopped'] and not z[2] else
                    'COLLISION_WITH_READY_STOP' if task['stopped'] and z[3] and task['collisions'] else 'OTHER')
                final[arm]=dict(label=task['safe_v16_label'],failure_type=failure,collisions=task['collisions'],
                    decisions=task['total_decisions'],stopped=task['stopped'],last_truth=z,last_state=last['predicted_state'],
                    first_collision_observation=None,collision_timing='Unavailable: source trace stores aggregate collisions only.')
                entries[arm]=dict(steps=steps,trace=trace,states=states,atoms=atoms)
            audit=prefix_audit(entries['ORIGINAL']['steps'],entries['EVENT']['steps'])
            if 'first_divergence' in audit or not audit['input_prefix_matched'] or not audit['action_prefix_matched']:
                raise ValueError('CAUSAL_PREFIX_MISMATCH')
            native_delta=max(native_delta,audit['max_logit_delta'])
            labels=[final[a]['label']=='PASS' for a in ARMS]
            outcome='both_pass' if all(labels) else 'loss' if labels[0] else 'win' if labels[1] else 'both_fail'
            case=dict(condition_id=ci,seed=seed,condition=cond,outcome=outcome,final=final,first=None,
                paths={a:str(p.relative_to(LINE)) for a,p in folders.items()})
            t=audit['first_method_action_difference']
            if t is not None:
                offset=t-cutoff; selected={a:entries[a]['steps'][offset] for a in ARMS}
                left,right=[selected[a] for a in ARMS]
                # Physical equality is checked before execution of the divergent action.
                if entries['ORIGINAL']['trace']['observations'][:t+1]!=entries['EVENT']['trace']['observations'][:t+1]:
                    raise ValueError('PHYSICAL_PREFIX_MISMATCH')
                truth=entries['ORIGINAL']['states'][t]
                predicted={a:selected[a]['predicted_state'] for a in ARMS}
                bits={a:[p>=.5 for p in predicted[a]] for a in ARMS}
                case['first']=dict(decision=t,offset=offset,input_key=left['raw']['input_key'],truth=truth,
                    instruction=left['raw']['instruction'],actions={a:selected[a]['executed_action'] for a in ARMS},
                    state=predicted,event={a:selected[a]['event_probabilities'] for a in ARMS},
                    method_logits={a:selected[a]['method_logits'] for a in ARMS},native_logits=left['logits'],
                    state_classes_same=bits['ORIGINAL']==bits['EVENT'],history_classes_same=bits['ORIGINAL'][:2]==bits['EVENT'][:2],
                    both_recognize_ready=all(predicted[a][3]>=.5 for a in ARMS),
                    terminal_truth=bool(entries['ORIGINAL']['atoms'][t]['terminal']),
                    anchor_truth=bool(entries['ORIGINAL']['atoms'][t]['anchor']),
                    state_threshold=.5,threshold_used_for_diagnosis_only=True)
                weights={a:nets[f'{a}_{seed}'].state_action.weight.detach().double() for a in ARMS}
                case['decomposition']=decomposition(left,right,weights)
                if outcome in ('loss','win'):
                    prefix=c.records(folders['ORIGINAL']/'PREFILL.jsonl')+entries['ORIGINAL']['steps'][:offset+1]
                    if len(prefix)!=t+1:raise ValueError('CAUSAL_REPLAY_LENGTH')
                    case['cpu_replay']=replay(prefix,{a:nets[f'{a}_{seed}'] for a in ARMS},cache,lookup,processed_inputs,left['logits'],selected)
                    if case['cpu_replay']['status']=='CPU_COMMON_PREFIX_REPRODUCED':replayed_steps+=2*len(prefix)
            elif outcome in ('loss','win'):
                raise ValueError('OUTCOME_DIFF_WITHOUT_ACTION_DIFF')
            cases.append(case)
    if len(cases)!=192:raise ValueError('MAIN_PAIR_DENOMINATOR')
    expected=next(p for p in read(run/'RESULT.json')['pairs'] if p['endpoint']=='main' and p['house'] is None and p['seed'] is None)
    if {(r['condition_id'],r['seed']) for r in cases if r['outcome']=='loss'}!={tuple(x) for x in expected['losses']}:
        raise ValueError('LOSS_PAIR_SET_CHANGED')
    after={tag:c.model_identity(net)['sha256'] for tag,net in nets.items()}
    if before!=after or torch.cuda.is_initialized():raise ValueError('DIAGNOSIS_MUTATION_OR_GPU')
    summary=summarize(cases)
    immutable(out/'CASES.json',cases)
    immutable(out/'RESULT.json',dict(status='READ_ONLY_FIRST_DIVERGENCE_LOCALIZATION_COMPLETE',main_pairs=192,
        summary=summary,models_unchanged=before==after,model_states=after,native_prefix_max_delta=native_delta,
        cpu_causal_step_forwards=replayed_steps,new_environment_actions=0,new_optimizer_updates=0,new_base_forwards=0,
        gpu_hours=0,seconds=time.monotonic()-began,
        limitation='All 192 main pairs reported; local score/component swaps do not execute counterfactual rollouts. Only fully cached, action/logit-verified shared histories enter actor/memory swap counts. Exposed one-house data; variants/seeds are correlated.'))
    immutable(out/'INPUT_MANIFEST.json',dict(files=manifest,source_run=str(run),all_source_group_files_verified=True))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUNTIME/'runs/event_001');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.run.resolve(),args.output.resolve())
