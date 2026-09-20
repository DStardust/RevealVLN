"""Build bounded proposals from real components; never fabricate certificates."""
import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RUNTIME, RESEARCH, c, load


def main():
    run=HERE/'raw_run_007';config=c.read(run/'CONFIG.json');result=c.read(run/'RESULT.json')
    sys.path.insert(0,str(RUNTIME))
    bridge=load('v15_assembly_core',RUNTIME/'core_bridge.py')
    factory=bridge.factory;Compiler=bridge.Compiler
    def inverse(actions):
        out=[]
        for a in reversed(actions):
            part=['L']*12+['F']+['R']*12 if a=='F' else ['R' if a=='L' else 'L']
            out=factory.compress(out+part)
        return out
    def closed(trace):
        a,b=trace['observations'][0]['pose'],trace['observations'][-1]['pose']
        return all(max(factory.pose_distance(x,y))<=1e-5 for x,y in
            [(a,b)]+[(a['sensors'][k],b['sensors'][k]) for k in a['sensors']])
    summaries=[];proposals=[]
    for house in config['houses']:
        summary=next(r for r in result['houses'] if r['house_id']==house['house_id'])
        rolemap={k:[r['mpcat40'],r['room']] for k,r in house['roles'].items()};first=next(iter(rolemap))
        compiler=Compiler(rolemap,{'probe':dict(anchor=first,terminal=first,instruction='Offline data audit')},house['expected_eligible'])
        records=[];corrections=[];counts=collections.Counter()
        for record in summary['records']:
            trace=c.read(LINE/record['path']);assert c.sha(LINE/record['path'])==record['sha256']
            if not compiler.complete(trace):continue
            geometrically_closed=closed(trace)
            if geometrically_closed!=record['closed_within_1e_5']:
                corrections.append(dict(path=record['path'],old_coefficient_test=record['closed_within_1e_5'],
                    physical_rotation_distance_test=geometrically_closed,
                    reason='Use the frozen factory quaternion-sign-invariant pose distance; raw coordinates and thresholds unchanged.'))
            events=compiler.atoms(trace['observations'])
            seen={r for e in events for r,ids in e.items() if ids}
            records.append(dict(record,trace=trace,events=events,seen=seen,geometrically_closed=geometrically_closed))
        retained=[];signatures=set()
        for hub in sorted({r['context']['hub'] for r in records}):
            rs=[r for r in records if r['context']['hub']==hub]
            tails=[r for r in rs if r['context']['phase']=='public_tail'];assert len(tails)==1
            visible=tails[0]['seen']
            loops=[r for r in rs if r['context']['phase']=='closed_return' and r['geometrically_closed']]
            outbounds=[r for r in rs if r['context']['phase']=='outbound']
            counts['actual_closed_return_components']+=len(loops)
            for la,lb in itertools.combinations(loops,2):
                for a in sorted(la['seen']-lb['seen']-visible):
                    for b in sorted(lb['seen']-la['seen']-visible):
                        if rolemap[a][1]==rolemap[b][1]:continue
                        for terminal in outbounds:
                            for t in sorted(terminal['seen']-{a,b}-visible):
                                cutoff=next(i for i,e in enumerate(terminal['events']) if e[t])
                                if any(e[a] or e[b] for e in terminal['events'][:cutoff+1]):continue
                                counts['history_and_terminal_core_combinations']+=1
                                for neutral in loops:
                                    if neutral['seen']&{a,b,t} or 'F' not in neutral['trace']['actions']:continue
                                    counts['with_observed_moving_neutral']+=1
                                    actions={'H_A':list(la['trace']['actions']),'H_B':list(lb['trace']['actions'])}
                                    fa,fb=(actions[h].count('F') for h in ('H_A','H_B'))
                                    fillers={}
                                    if fa!=fb:
                                        gap=abs(fa-fb);prefix=[]
                                        for action in neutral['trace']['actions']:
                                            prefix.append(action)
                                            if 2*prefix.count('F')==gap:break
                                        if 2*prefix.count('F')!=gap:counts['no_balancing_prefix']+=1;continue
                                        h='H_A' if fa<fb else 'H_B';filler=prefix+inverse(prefix)
                                        fillers[h]=[len(actions[h]),len(actions[h])+len(filler)]
                                        actions[h]+=filler
                                    ca,cb=(collections.Counter(actions[h]) for h in ('H_A','H_B'))
                                    if ca['L']-ca['R']!=cb['L']-cb['R']:
                                        counts['turn_count_mismatch']+=1;continue
                                    target=max(ca['L'],cb['L'])
                                    for h in actions:actions[h]+= ['L','R']*(target-collections.Counter(actions[h])['L'])
                                    neutral_actions=neutral['trace']['actions'];histories={};spans={}
                                    for h,base in actions.items():
                                        histories[h]=base+['L','R']*4
                                        histories[h+'_N']=base+neutral_actions+['L','R']*4
                                        spans[h+'_N']=[len(base),len(base)+len(neutral_actions)]
                                    tail=terminal['trace']['actions'][:cutoff]+['S']
                                    continuations={'C0':tail,'C_A':la['trace']['actions']+tail,'C_B':lb['trace']['actions']+tail}
                                    if max(map(len,histories.values()))+max(map(len,continuations.values()))>500:
                                        counts['over_500']+=1;continue
                                    signature=(hub,a,b,t,tuple(histories['H_A']),tuple(histories['H_B']))
                                    if signature in signatures:continue
                                    signatures.add(signature)
                                    other=next(r for r in rolemap if r not in (a,b,t))
                                    aliases={'anchor_A':a,'anchor_B':b,'terminal':t,'irrelevant':other}
                                    roles={k:house['roles'][r] for k,r in aliases.items()}
                                    def phrase(r):return 'the '+r['raw_match']['value']+' in the '+r['room']
                                    tasks={task:dict(anchor=anchor,terminal='terminal',instruction='First see '+phrase(roles[anchor])+' in two consecutive observations, then see '+phrase(roles['terminal'])+' in two consecutive observations, and stop immediately.') for task,anchor in [('task_A','anchor_A'),('task_B','anchor_B')]}
                                    sources=[x['path'] for x in (la,lb,terminal,neutral)]
                                    family_id='V15_NEW_'+hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
                                    family=dict(family_id=family_id,house=house['house_id'],partition=house['partition'],
                                        roles=roles,scene=house['scene'],assets=house['assets'],
                                        compiler=dict(roles={k:rolemap[r] for k,r in aliases.items()},tasks=tasks,
                                            eligible={k:house['expected_eligible'][r] for k,r in aliases.items()}),
                                        candidate=dict(position=la['trace']['observations'][0]['pose']['position'],yaw_bin=0,
                                            histories=histories,continuations=continuations),neutral_span=spans,
                                        balancing_filler_spans=fillers,source_components=sources,
                                        source_component_sha256={p:c.sha(LINE/p) for p in sources},
                                        task_terminal_only=dict(terminal='terminal',instruction='See '+phrase(roles['terminal'])+' in two consecutive observations, then stop. No earlier object sighting is required.'),
                                        exact_motion_counts_matched=True,training_admission=False,physical_family_verified=False,
                                        next_required='Native initial-state proposal, complete raw physical crossed replay, all neutral/filler checks and independent raw-array audit.')
                                    retained.append(family)
        retained.sort(key=lambda f:f['family_id'])
        proposals.extend(retained[:3])
        summaries.append(dict(house=house['house_id'],partition=house['partition'],counts=dict(counts),
            bounded_selected_proposals=min(3,len(retained)),distinct_proposals=len(retained),
            pose_summary_corrections=corrections,physical_family_certificates=0))
        print(summaries[-1],flush=True)
    c.write(RESEARCH/'NEW_HOUSE_ASSEMBLY.json',dict(families=proposals,summaries=summaries,
        status='REAL_COMPONENT_PROPOSALS_ONLY',training_admission=False,source_sha256=c.sha(Path(__file__))),True)


if __name__=='__main__':main()
