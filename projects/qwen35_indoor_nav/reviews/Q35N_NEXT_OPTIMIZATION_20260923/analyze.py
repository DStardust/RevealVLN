"""CPU-only diagnosis of immutable paired rollouts and newly sealed training data."""
import collections,hashlib,json,statistics,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];SFT=LINE/'sft_acceptance'
FILES={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):FILES[str(p)]=sha(p);return json.loads(p.read_text())
def rows(p):FILES[str(p)]=sha(p);return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(name,v):(HERE/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def episode(e):
    distances=e['distances'];before=distances[:-1];reached=any(d<3 for d in before)
    return dict(episode_id=e['episode_id'],house=e['house'],success=e['success'],stopped=e['stopped'],steps=e['steps'],collisions=e['collisions'],
        reached_before_decision=reached,oracle_success=e['oracle_success'],legal_stop_opportunities=sum(d<3 for d in before),
        late_arrival_without_stop_budget=(not reached and distances[-1]<3),
        final_distance=distances[-1],minimum_distance=min(distances),failed_stop=e['stopped'] and not e['success'])
def compare(name,run):
    folder=SFT/name/'runs'/run;result=read(folder/'REVIEW.json');pairs=[]
    for f in sorted(folder.glob('sessions/*/pairs/*/PAIR.json')):
        r=read(f)
        for file,h in r['trace_hashes'].items():assert sha(f.parent/file)==h,'SEALED_TRACE_CHANGED'
        assert r['state_unchanged'] and r['input_prefix_matched'] and r['base_action_prefix_matched']
        pairs.append(r)
    assert len(pairs)==100 and len({r['rank'] for r in pairs})==100
    arms={}
    for arm in ('A','B'):
        es=[episode(r[arm]) for r in pairs];stops=[e for e in es if e['failed_stop']];bins={'3_to_6':0,'6_to_10':0,'10_plus':0}
        for e in stops:bins['3_to_6' if e['final_distance']<6 else '6_to_10' if e['final_distance']<10 else '10_plus']+=1
        arms[arm]=dict(successes=sum(e['success'] for e in es),reached=sum(e['oracle_success'] for e in es),missed_legal_stop=sum(e['reached_before_decision'] and not e['success'] for e in es),failed_stop_count=len(stops),failed_stop_distance_bins=bins,
            steps=sum(e['steps'] for e in es),collisions=sum(e['collisions'] for e in es),budget_exhausted=sum(not e['stopped'] for e in es),episodes=es)
    lost_reach=[r['episode_id'] for r in pairs if r['A']['oracle_success'] and not r['B']['oracle_success']]
    gained_reach=[r['episode_id'] for r in pairs if not r['A']['oracle_success'] and r['B']['oracle_success']]
    return dict(n=100,summary=result,arms=arms,lost_reach=lost_reach,gained_reach=gained_reach,
        interpretation='OSR is reachability on actual terminated rollouts, not an upper bound after changing STOP or motion; different stop points censor trajectory lengths.')
def collection():
    run=SFT/'ordinary_stop_coverage_v15/runs/coverage_001';status=read(run/'STATUS.json');order=read(SFT/'ordinary_stop_coverage_v15/manifests/FIT_ORDER.json');counts=collections.Counter();by_mode=collections.defaultdict(collections.Counter);houses=collections.Counter();seen={};completed=[];teacher_motion=[];pertrajectory=[]
    for f in sorted(run.glob('collect/sessions/*/pairs/*/COLLECT.json')):
        seal=read(f);folder=f.parent/'C';assert seal['rank'] not in completed;completed.append(seal['rank'])
        for file,h in seal['files'].items():assert sha(folder/file)==h,'COLLECTION_FILE_CHANGED'
        assert seal['base_unchanged'];mode=seal['mode'];houses[seal['house']]+=1
        labels=rows(folder/'SUPERVISION_ONLY.jsonl');policy=rows(folder/'POLICY_STEPS.jsonl');assert len(labels)==len(policy)==seal['steps'];counter=collections.Counter()
        for i,(label,pol) in enumerate(zip(labels,policy)):
            assert label['step']==pol['step']==i+1 and label['executed_action']==pol['executed_action']
            d=label['distance_before'];assert label['target']==int(d<3);key=pol['raw']['input_key'];target=label['target'];counts['observations']+=1;counts['positive']+=target;by_mode[mode]['observations']+=1;by_mode[mode]['positive']+=target
            seen.setdefault(key,set()).add(target)
            bucket='near_3_to_6' if 3<=d<6 else 'far_ge6' if d>=6 else 'inside_lt3';by_mode[mode][bucket]+=1
            if mode=='TEACHER' and d>=3 and label['executed_action']!='STOP':
                counter['motion_labels']+=1;actual=label['executed_action'];proposal=pol['proposed_action'];moves=['move_forward','turn_left','turn_right'];best_motion=moves[max(range(3),key=lambda k:pol['logits'][k])]
                counter['proposed_premature_stop']+=int(proposal=='STOP');counter['motion_teacher_disagreement']+=int(best_motion!=actual)
                teacher_motion.append(dict(admission='PARTIAL_TRAIN_ONLY_DEBUG_INDEX',rank=seal['rank'],house=seal['house'],step=i+1,input_key=key,feature_path=str(folder/'FEATURES.pt'),feature_index=i,actual_teacher_motion=actual,reference_proposal=proposal,reference_best_motion=best_motion,teacher_motion_disagreement=best_motion!=actual,distance_label_file=str(folder/'SUPERVISION_ONLY.jsonl'),policy_file=str(folder/'POLICY_STEPS.jsonl'),seal=str(f)))
        pertrajectory.append(dict(rank=seal['rank'],house=seal['house'],mode=mode,steps=seal['steps'],positive=seal['positive'],counts=dict(counter)))
    totals=sum((collections.Counter(x['counts']) for x in pertrajectory),collections.Counter());eligible=[x for x in pertrajectory if x['counts'].get('motion_labels')]
    macro=statistics.mean(x['counts'].get('motion_teacher_disagreement',0)/x['counts']['motion_labels'] for x in eligible) if eligible else None
    (HERE/'TEACHER_MOTION_INDEX.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in teacher_motion))
    failure=run/'collect/sessions/gpu4_1790090483015412981/pairs/pair_279/C'
    return dict(status=status,planned=640,sealed=len(completed),missing_ranks=sorted(set(range(640))-set(completed)),houses=dict(houses),unique_houses=len(houses),counts=dict(counts),unique_inputs=len(seen),conflicting_input_keys=[k for k,v in seen.items() if len(v)>1],by_mode={k:dict(v) for k,v in by_mode.items()},teacher_motion=dict(totals),trajectory_mean_motion_disagreement=macro,pertrajectory=pertrajectory,failure_start=rows(failure/'EPISODES.jsonl'),failure_detail=rows(failure/'SERVICE_FAILURES.jsonl'),label_caveat='Disagreement with one actual reference-route teacher, not proof that every different motion is invalid; teacher-only outside3m. No new forward/optimizer/rollout.',recovery_data_gap='POLICY and TEACHER start independently from the initial state. No teacher takeover at the actual policy error state; current train.py uses only binary STOP labels.')
def longrun():
    root=LINE/'research/continuation_memory_v1/evidence_state_policy_v1/expanded_longtrain_v1/runs/long_001';result=[]
    for p in sorted(root.glob('evaluations/STEP_*/RESULT.json')):
        r=read(p);result.append(dict(checkpoint=p.parent.name,complete=r['complete'],planned=r['planned'],metrics=[m for m in r['metrics'] if m['house'] is None]))
    return dict(status=read(root/'STATUS.json'),checkpoints=result,scope='Task-conditioned recurrent memory, zero base updates; these are not ordinary R2R SR measurements.')
if __name__=='__main__':
    result=dict(unix=time.time(),new_gpu_forwards=0,new_navigation_episodes=0,new_optimizer_updates=0,V13=compare('ordinary_stop_refit_v13','stop_002'),V14=compare('ordinary_stop_boundary_v14','boundary_001'),V15=collection(),memory_longrun=longrun())
    write('ANALYSIS.json',result);write('READ_EVIDENCE.json',dict(files=FILES,script_sha256=sha(Path(__file__)),scope='New CPU reanalysis of existing immutable logs; no new efficacy experiment'))
    print(json.dumps(dict(V13={k:{j:v for j,v in r.items() if j!='episodes'} for k,r in result['V13']['arms'].items()},V15={k:v for k,v in result['V15'].items() if k not in ('missing_ranks','pertrajectory','failure_detail')},memory_status=result['memory_longrun']['status']),ensure_ascii=False,indent=2))
