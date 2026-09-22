"""Read-only ORIGINAL recovery localization: real causal CPU forwards and sealed live logs."""
import argparse
from collections import Counter,defaultdict
import math
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
RUNTIME=BASE/'teacher_alignment_v1'
sys.path.insert(0,str(RUNTIME))
from shared import c,read,write,sha,digest,immutable,load,LINE,CPU,CONTROL,V16,make_head
from evaluate_continuations import registry,admitted
from evaluator_v16 import legacy,state_sequence,evaluate
from select_action import select

def confusion(values):
    counts=Counter();brier=0.
    for p,y in values:
        if not math.isfinite(p) or not 0<=p<=1 or y not in (0,1):raise ValueError('PROBABILITY_OR_TRUTH')
        counts['tp' if p>=.5 and y else 'fp' if p>=.5 else 'fn' if y else 'tn']+=1
        brier+=(p-y)**2
    return dict(n=len(values),**{k:counts[k] for k in ('tp','fp','fn','tn')},brier=brier/len(values) if values else None)

def history_origin(states,truth,events,event_truth,cutoff):
    """Threshold is diagnostic only. At t=0 no prior SEE2 anchor can have happened."""
    if not (len(states)==len(truth)==len(events)==len(event_truth)) or not 0<=cutoff<len(states):raise ValueError('SEQUENCE_ALIGNMENT')
    prior=states[0][0];cross=next((t for t in range(cutoff+1) if not truth[t][1] and states[t][1]>=.5),None)
    seen=[t for t in range(cutoff+1) if event_truth[t][0]]
    detected=[t for t in seen if events[t][0]>=.5]
    false_at_cut=not truth[cutoff][1] and states[cutoff][1]>=.5
    return dict(initial_prior=prior,initial_prior_false_positive=prior>=.5,
        false_history_at_cutoff=false_at_cut,first_false_crossing=cross,
        false_history_origin=('initial_prior' if prior>=.5 else 'event_accumulation') if false_at_cut else None,
        true_anchor_witness_steps=seen,first_witness=seen[0] if seen else None,
        first_witness_detected=bool(seen and events[seen[0]][0]>=.5),any_true_witness_detected=bool(detected),
        history_missing_at_cutoff=bool(truth[cutoff][1] and states[cutoff][1]<.5),
        detected_then_lost=bool(detected and any(states[t][1]<.5 for t in range(detected[0],cutoff+1))),
        oldest_event_gap=cutoff-seen[0] if seen else None,
        takeover_truth=truth[cutoff],takeover_prediction=states[cutoff])

def stage_failure(task,truth,probability):
    if task['safe_v16_label']=='PASS':return 'PASS'
    if not task['stopped']:return 'BUDGET_EXHAUSTED' if task['budget_exhausted'] else 'OTHER_NONSTOP'
    if not truth[0]:return 'STOP_MISSING_HISTORY_WITH_FALSE_BELIEF' if probability[0]>=.5 else 'STOP_DESPITE_CORRECT_MISSING_HISTORY'
    if not truth[2]:return 'STOP_MISSING_TERMINAL_WITH_FALSE_DETECTION' if probability[2]>=.5 else 'STOP_DESPITE_CORRECT_MISSING_TERMINAL'
    return 'READY_STOP_WITH_COLLISION' if task['collisions'] else 'OTHER'

def event_summary(cases):
    return dict(n=len(cases),false_history_at_cutoff=sum(r['false_history_at_cutoff'] for r in cases),
        initial_prior_false_positives=sum(r['initial_prior_false_positive'] for r in cases),
        false_history_origins=dict(Counter(r['false_history_origin'] for r in cases if r['false_history_origin'])),
        true_history_n=sum(bool(r['takeover_truth'][1]) for r in cases),
        missing_history_n=sum(not r['takeover_truth'][1] for r in cases),
        missed_history_at_cutoff=sum(r['history_missing_at_cutoff'] for r in cases),
        witnessed_n=sum(r['first_witness'] is not None for r in cases),
        first_witness_detected=sum(r['first_witness_detected'] for r in cases),
        any_witness_detected=sum(r['any_true_witness_detected'] for r in cases),
        detected_then_lost=sum(r['detected_then_lost'] for r in cases),
        teacher_action_disagreements=sum(r['action']!=r['teacher_action'] for r in cases),
        correct_state_teacher_disagreements=sum(r['state_correct'] and r['action']!=r['teacher_action'] for r in cases))

def probe(run,out,manifest):
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    data=read(CPU/'DATA.json');cache_path=CONTROL/'features/FEATURES.pt';cache_record=read(CONTROL/'features/FEATURE_RESULT.json')
    if sha(cache_path)!=cache_record['file_sha256']:raise ValueError('CACHE_HASH')
    cache={k:v.float() for k,v in torch.load(cache_path,map_location='cpu',weights_only=True).items()}
    manifest[str(cache_path.relative_to(LINE))]=cache_record['file_sha256'];manifest[str((CPU/'DATA.json').relative_to(LINE))]=sha(CPU/'DATA.json')
    families={f['family_id']:f for f in data['raw_families']};cases=[];counts=Counter();unique={};event_records=[];models=[]
    for seed in (1209,1210,1211):
        folder=run/'train'/f'ORIGINAL_{seed}';record=read(folder/'RESULT.json');net=make_head(f'ORIGINAL_{seed}').eval()
        if sha(folder/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('HEAD_HASH')
        net.load_state_dict(torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True));before=c.model_identity(net)['sha256']
        if before!=record['final']:raise ValueError('HEAD_STATE')
        manifest[str((folder/'FINAL.pt').relative_to(LINE))]=record['checkpoint_sha256']
        with torch.inference_mode():
            for family in data['families']:
                rows=[r for r in family['sequences'] if any(r['action_masks'])]
                if len(rows)!=8:raise ValueError('TEACHER_SCOPE')
                lengths=[len(r['features']) for r in rows];maximum=max(lengths)
                ids=torch.tensor([r['features']+[0]*(maximum-n) for r,n in zip(rows,lengths)])
                alive=torch.tensor([[True]*n+[False]*(maximum-n) for n in lengths])
                value=net(cache['features'][ids],cache['logits'][ids],alive)
                z=value['state'].tolist();ev=value['event_logits'].sigmoid().tolist();logits=value['logits'].tolist()
                for i,row in enumerate(rows):
                    n=lengths[i];raw=families[family['family_id']];compiler=legacy.Compiler(**raw['compiler'])
                    tracepath=LINE/row['source_trace']['path']
                    if str(tracepath.relative_to(LINE)) not in manifest:
                        if sha(tracepath)!=row['source_trace']['sha256']:raise ValueError('TRACE_HASH')
                        manifest[str(tracepath.relative_to(LINE))]=row['source_trace']['sha256']
                    trace=read(tracepath);truth=state_sequence(compiler,trace['observations'],row['task']);atoms=compiler.atoms(trace['observations'])
                    if truth!=row['state_targets']:raise ValueError('TRUTH_ALIGNMENT')
                    if row['task']=='task_A':
                        info=history_origin(z[i][:n],truth,ev[i][:n],row['event_targets'],row['cutoff'])
                        t=row['cutoff'];pred=z[i][t]
                        cases.append(dict(seed=seed,split=family['split'],family=family['family_id'],house=family['house'],
                            history=row['history'],stratum=family['stratum'],**info,
                            action=max(range(4),key=lambda k:logits[i][t][k]),teacher_action=row['targets'][t],
                            state_correct=all((p>=.5)==bool(y) for p,y in zip(pred,truth[t])),
                            first_witness_event_probability=ev[i][info['first_witness']][0] if info['first_witness'] is not None else None,
                            cutoff_stop_margin=logits[i][t][3]-max(logits[i][t][:3]),trace=row['source_trace']))
                    for t in range(n):
                        for bit,role in [(0,'anchor'),(1,'terminal')]:
                            if not row['event_masks'][t][bit]:continue
                            label=row['event_targets'][t][bit];key=(seed,row['features'][t],bit)
                            pixels=0
                            if t and atoms[t][role]:
                                pixels=max(min(trace['observations'][t]['pixels'].get(str(idx),0),trace['observations'][t-1]['pixels'].get(str(idx),0)) for idx in atoms[t][role])
                            point=dict(seed=seed,split=family['split'],feature=row['features'][t],role=role,
                                probability=ev[i][t][bit],truth=label,positive_scale='near_256_1023' if pixels and pixels<1024 else 'clear_1024plus' if pixels else 'absent')
                            if key in unique:
                                old=unique[key]
                                if old['truth']!=label or old['split']!=family['split']:raise ValueError('FEATURE_LABEL_OR_SPLIT_CONFLICT')
                                if abs(old['probability']-point['probability'])>1e-6:raise ValueError('EVENT_HEAD_NOT_POINTWISE')
                            else:unique[key]=point
                    counts['sequence_forwards']+=1;counts['causal_step_forwards']+=n
                write(out/'STATUS.json',dict(status='RUNNING',stage='causal_cpu_probe',seed=seed,family=family['family_id'],**dict(counts),gpu_hours=0,new_optimizer_updates=0))
        after=c.model_identity(net)['sha256']
        if before!=after or torch.cuda.is_initialized():raise ValueError('CPU_PROBE_MUTATION_OR_GPU')
        models.append(dict(seed=seed,initial_state_sha256=before,final_state_sha256=after,unchanged=True))
        print('causal_probe',seed,dict(counts),flush=True)
    summary=[];calibration=[]
    for split in ('FIT','DEV'):
        for seed in (None,1209,1210,1211):
            for history in ('all','seen','missing'):
                selected=[r for r in cases if r['split']==split and (seed is None or r['seed']==seed) and (history=='all' or r['history'].startswith(history))]
                summary.append(dict(split=split,seed=seed,history=history,**event_summary(selected)))
            for role in ('anchor','terminal'):
                for scale in ('all','absent','near_256_1023','clear_1024plus'):
                    selected=[r for r in unique.values() if r['split']==split and (seed is None or r['seed']==seed) and r['role']==role and (scale=='all' or r['positive_scale']==scale)]
                    calibration.append(dict(split=split,seed=seed,role=role,scale=scale,**confusion([(r['probability'],r['truth']) for r in selected])))
    immutable(out/'CAUSAL_PROBE.json',dict(counts=dict(counts),models=models,summary=summary,cases=cases,
        event_calibration=calibration,unique_feature_event_records=len(unique),
        scope='Original selected actual teacher trajectories only; FIT and exposed DEV reported separately. Full causal CPU recurrence over frozen cached Qwen features. No optimizer or new Qwen forward.',
        limitation='Pointwise event records are deduplicated by input feature, but remain correlated within houses/families. Memory probe is conditional on recorded teacher paths; it is not new navigation success.'))
    return cases

def live(run,out,manifest,probe_cases):
    reg=registry(run);groups=admitted(run,reg)
    if len(groups)!=128 or read(run/'RESULT.json')['complete']!=768:raise ValueError('WAIT_FOR_COMPLETE_REGISTERED_EVALUATION')
    raw={f['family_id']:f for f in read(run/'DATA.json')['raw_families']};episodes=[];prefixes=[];references={(r['seed'],r['family'],r['history']):r for r in probe_cases if r['split']=='DEV'}
    for slot in reg['slots']:
        cond=reg['conditions'][slot['condition']]
        if slot['arm']!='ORIGINAL' or cond['task_id']!='task_A':continue
        session=groups[slot['condition']];folder=session/'rollouts'/f"{slot['rank']:04d}"
        f=raw[cond['family_id']];trace=read(folder/'TRACE_PRIVILEGED.json');steps=c.records(folder/'POLICY_STEPS.jsonl');task=read(folder/'TASK_RESULT.json')
        cut=len(f['histories'][cond['history_id']]);compiler=legacy.Compiler(**f['compiler']);states=state_sequence(compiler,trace['observations'],'task_A');atoms=compiler.atoms(trace['observations'])
        checked=evaluate(compiler,trace,'task_A',cut)
        if any(task[k]!=v for k,v in checked.items()):raise ValueError('TASK_RECOMPUTATION')
        if [r['decision'] for r in steps]!=list(range(cut,len(trace['actions']))):raise ValueError('POLICY_TRACE_OFFSET')
        ready_recognized=[];witnesses=[];false_crossings=[];prior=None
        for row in steps:
            t=row['decision'];truth=states[t];pred=row['predicted_state']
            if select(row['logits'],row['method_logits'])['executed_action']!=row['executed_action']:raise ValueError('ACTION_RECOMPUTATION')
            if row['executed_action']!=c.ACTIONS['FLRS'.index(trace['actions'][t])]:raise ValueError('ACTUAL_ACTION_MISMATCH')
            if truth[3] and pred[3]>=.5 and row['executed_action']!='STOP':ready_recognized.append(t)
            if atoms[t]['anchor']:
                witnesses.append(dict(step=t,event_probability=row['event_probabilities'][0],history_probability=pred[1]))
            if not truth[1] and pred[1]>=.5 and (prior is None or prior[1]<.5):false_crossings.append(t)
            prior=pred
        end=steps[-1];truth=states[end['decision']];take=steps[0];reference=references[slot['seed'],cond['family_id'],cond['history_id']]
        delta=max(abs(a-b) for a,b in zip(take['predicted_state'],reference['takeover_prediction']))
        prefix_agree=all((a>=.5)==(b>=.5) for a,b in zip(take['predicted_state'],reference['takeover_prediction']))
        action_match=take['executed_action']==c.ACTIONS[reference['action']]
        row=dict(slot,**cond,status=task['safe_v16_label'],failure_type=stage_failure(task,truth,end['predicted_state']),
            collisions=task['collisions'],exhausted=task['budget_exhausted'],decisions=task['total_decisions'],autonomous_decisions=len(steps),
            takeover_truth=states[cut],takeover_prediction=take['predicted_state'],takeover_action=take['executed_action'],
            final_truth=truth,final_prediction=end['predicted_state'],final_action=end['executed_action'],
            first_autonomous_anchor=witnesses[0] if witnesses else None,
            ready_recognized_but_continued=ready_recognized,
            false_history_crossings=false_crossings,cached_cpu_live_state_delta=delta,
            cached_cpu_live_state_classes_equal=prefix_agree,cached_cpu_live_action_equal=action_match,
            first_forward=next((r['decision'] for r in steps if r['executed_action']=='move_forward'),None),
            path=str(folder.relative_to(LINE)),
            offline_history_origin=reference['false_history_origin'] if prefix_agree else 'CACHE_LIVE_CLASSIFICATION_DISAGREES')
        episodes.append(row)
        for name in (f'GROUP_{slot["condition"]:03d}.json',f'STATE_SEAL_{slot["condition"]:03d}.json'):
            p=session/name;manifest[str(p.relative_to(LINE))]=sha(p)
    summaries=[]
    for history in ('all','seen','missing'):
        for seed in (None,1209,1210,1211):
            selected=[r for r in episodes if (history=='all' or r['history_id'].startswith(history)) and (seed is None or r['seed']==seed)]
            summaries.append(dict(history=history,seed=seed,n=len(selected),passed=sum(r['status']=='PASS' for r in selected),
                failure_types=dict(Counter(r['failure_type'] for r in selected)),
                takeover_history=confusion([(r['takeover_prediction'][0],r['takeover_truth'][0]) for r in selected]),
                took_forward=sum(r['first_forward'] is not None for r in selected),colliding=sum(r['collisions']>0 for r in selected),
                ever_ready_recognized_but_continued=sum(bool(r['ready_recognized_but_continued']) for r in selected),
                ever_autonomous_anchor_witness=sum(r['first_autonomous_anchor'] is not None for r in selected),
                ever_autonomous_anchor_first_detected=sum(r['first_autonomous_anchor'] is not None and r['first_autonomous_anchor']['event_probability']>=.5 for r in selected),
                false_history_origins=dict(Counter(r['offline_history_origin'] for r in selected if r['offline_history_origin'])),
                immediate_stop=sum(r['autonomous_decisions']==1 and r['final_action']=='STOP' for r in selected)))
    immutable(out/'ORIGINAL_RECOVERY.json',dict(episodes=episodes,summary=summaries,
        cached_cpu_live_state_max_delta=max(r['cached_cpu_live_state_delta'] for r in episodes),
        cached_cpu_live_state_class_mismatches=sum(not r['cached_cpu_live_state_classes_equal'] for r in episodes),
        cached_cpu_live_action_mismatches=sum(not r['cached_cpu_live_action_equal'] for r in episodes),
        note='Only takeover compares the same input; post-action observations are policy-dependent. Diagnostic0.5 threshold is not deployed. No oracle enters actor; future query is unused.'))
    return summaries

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUNTIME/'runs/aligned_001');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();run=args.run
    sources=read(run/'SOURCE_LOCK.json')['files'];manifest={}
    for p in [BASE/'model.py',V16/'evaluator_v16.py',LINE/'data_pipeline/mechanism_factory_v2/compiler.py',RUNTIME/'shared.py',RUNTIME/'evaluate_continuations.py']:
        rel=str(p.relative_to(LINE))
        if sha(p)!=sources[rel]:raise ValueError('FROZEN_DEPENDENCY_CHANGED')
        manifest[rel]=sources[rel]
    for name in ['RESULT.json','PROTOCOL.json','DATA.json','EVALUATION_REGISTRY.json']:
        p=run/name;manifest[str(p.relative_to(LINE))]=sha(p)
    write(out/'STATUS.json',dict(status='RUNNING',stage='causal_cpu_probe',gpu_hours=0,new_optimizer_updates=0))
    try:
        cases=probe(run,out,manifest);summaries=live(run,out,manifest,cases)
        immutable(out/'INPUT_MANIFEST.json',dict(files=manifest,source_sha256=sha(Path(__file__)),source_run=str(run),sealed_group_files_verified=True))
        immutable(out/'RESULT.json',dict(status='READ_ONLY_RECOVERY_LOCALIZATION_COMPLETE',models=3,teacher_sequence_forwards=1800,
            main_rollouts=192,gpu_hours=0,base_forwards=0,optimizer_updates=0,new_environment_actions=0,
            seconds=time.monotonic()-start,live_summary=summaries,method_benefit_measured=False))
        write(out/'STATUS.json',dict(status='COMPLETE',stage='diagnosis_complete',seconds=time.monotonic()-start,gpu_hours=0))
    except BaseException as exc:
        write(out/'STATUS.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start,gpu_hours=0));raise

if __name__=='__main__':main()
