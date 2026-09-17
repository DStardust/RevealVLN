"""Pure CPU witness-bank proposal API. Proposed compositions require real replay."""
import collections
import copy
import importlib.util
import itertools
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('bank_witness_compatibility',HERE.parent/'compatibility_cpu/audit.py')
compat=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(compat)

def role_key(signature):return json.dumps(list(signature),separators=(',',':'))

def normalize_roles(rows):
    result={};used=set()
    for row in rows:
        if set(row)!={'signature','eligible_ids'}:raise ValueError('ROLE_SCHEMA')
        sig=row['signature'];ids=row['eligible_ids']
        if not isinstance(sig,(list,tuple)) or len(sig)!=3:raise ValueError('ROLE_SIGNATURE')
        cat,room,raw=sig
        if room not in compat.planner.INDOOR_ROOMS or raw not in compat.planner.RAW.get(cat,set()):raise ValueError('OUTSIDE_FROZEN_VOCABULARY')
        if not ids or any(type(i) is not int or i in compat.planner.RESERVED or i<0 for i in ids):raise ValueError('ROLE_IDS')
        if len(set(ids))!=len(ids) or used.intersection(ids):raise ValueError('OVERLAPPING_ROLE_IDS')
        used.update(ids);key=role_key(sig)
        if key in result:raise ValueError('DUPLICATE_ROLE')
        result[key]=sorted(ids)
    if not result:raise ValueError('EMPTY_ROLE_CATALOG')
    return result

def normalize_bank(records):
    groups=collections.defaultdict(list);seen_ids=set();house_roles={};count=0
    for record in records:
        count+=1
        if record.get('split')!='FIT':raise ValueError('ONLY_FIT_BANK_ALLOWED')
        for field in ('id','house_id','scene_fingerprint'):
            if not isinstance(record.get(field),str) or not record[field]:raise ValueError('MISSING_IDENTITY:'+field)
        if record['id'] in seen_ids:raise ValueError('DUPLICATE_TRACE_ID')
        seen_ids.add(record['id'])
        if type(record.get('closed_loop')) is not bool:raise ValueError('CLOSED_LOOP_FLAG_TYPE')
        trace=record['trace']
        if not compat.compiler.complete(trace):raise ValueError('UNKNOWN_OR_INCOMPLETE_TRACE_NOT_NEGATIVE')
        if any(a not in ('F','L','R') for a in trace['actions']):raise ValueError('BANK_COMPONENTS_MOTION_ONLY')
        if 'trace_hash' in trace and compat.digest({k:v for k,v in trace.items() if k!='trace_hash'})!=trace['trace_hash']:
            raise ValueError('TRACE_SELF_HASH_MISMATCH')
        registry=normalize_roles(record['roles'])
        house=(record['house_id'],record['scene_fingerprint'])
        if house in house_roles and house_roles[house]!=registry:raise ValueError('INCONSISTENT_WHOLE_HOUSE_ROLE_CATALOG')
        house_roles[house]=registry
        first_obs=trace['observations'][0]
        pose=first_obs['pose']
        # A caller's closure flag alone cannot bypass observed pose discrepancy.
        if record['closed_loop'] and (not trace['actions'] or not compat.closed_pose(pose,trace['observations'][-1]['pose'])):
            raise ValueError('CLAIMED_LOOP_NOT_CLOSED_IN_STORED_POSES')
        key=compat.digest(dict(house=house,pose=pose,rgb=first_obs['rgb_hash'],semantic=first_obs['semantic_hash']))
        compiler_roles={k:tuple(json.loads(k)[:2]) for k in registry}
        role=next(iter(registry))
        compiler=compat.Compiler(compiler_roles,{'probe':dict(anchor=role,terminal=role,instruction='CPU witness extraction')},registry)
        events=compiler.atoms(trace['observations']);first={}
        for step,event in enumerate(events):
            for role,ids in event.items():
                if ids is None:raise ValueError('UNKNOWN_ROLE_EVENT')
                if ids:first.setdefault(role,(step,min(ids)))
        groups[key].append(dict(id=record['id'],house_id=record['house_id'],scene_fingerprint=record['scene_fingerprint'],
            hub_key=key,hub_pose=copy.deepcopy(pose),registry=registry,role_first=first,
            closed_loop=record['closed_loop'],actions=list(trace['actions']),
            trace_digest=compat.digest(trace),trace_ref=record.get('trace_ref',record['id']),
            public_tail=trace['actions']==list('LRLRLRLR')))
    for rows in groups.values():rows.sort(key=lambda row:row['id'])
    return groups,count

def component(rows,required,forbidden=(),*,closed=False,prefer_movement=False):
    choices=[]
    for row in rows:
        if required not in row['role_first']:continue
        cutoff=row['role_first'][required][0]
        if closed:
            if not row['closed_loop'] or len(row['actions'])>504:continue
            cutoff=len(row['actions'])
        if any(role in row['role_first'] and row['role_first'][role][0]<=cutoff for role in forbidden):continue
        actions=row['actions'][:cutoff]
        choices.append(dict(trace_id=row['id'],trace_ref=row['trace_ref'],trace_digest=row['trace_digest'],
            actions=actions,cutoff_observation=cutoff,required_role=required,
            event_step=row['role_first'][required][0],event_instance=row['role_first'][required][1],
            forbidden_roles=list(forbidden),stored_closed_loop_checked=closed,
            forward_actions=actions.count('F')))
    if not choices:return None
    return min(choices,key=lambda c:(-(c['forward_actions']>=2) if prefer_movement else 0,len(c['actions']),c['trace_id']))

def count_profile(actions):return {key:actions.count(key) for key in ('F','L','R')}

def propose(records,max_programs=64,require_closed_anchors=True,min_irrelevant_forward=0):
    """Return deterministic component proposals; no simulation, mutation, or certification.

    Defaults require actual stored A/B closed loops. I prefers >=2F but remains
    explicitly unresolved if no closed I loop exists. Set min_irrelevant_forward=2
    to exclude turn-only I loops; never silently relabel them useful detours.
    """
    if type(max_programs) is not int or not 1<=max_programs<=1000:raise ValueError('PROGRAM_LIMIT')
    if type(require_closed_anchors) is not bool or type(min_irrelevant_forward) is not int or min_irrelevant_forward<0:
        raise ValueError('FILTER_ARGUMENTS')
    groups,n=normalize_bank(records);proposals=[];enumerated=0;pair_count=0
    for hub,rows in sorted(groups.items()):
        role_names=sorted({r for row in rows for r in row['role_first']})
        for a,b in itertools.combinations(role_names,2):
            if not compat.semantic_distinct((a,b)):continue
            wa=component(rows,a,[b],closed=require_closed_anchors)
            wb=component(rows,b,[a],closed=require_closed_anchors)
            if not wa or not wb:continue
            pair_count+=1
            for terminal in role_names:
                if not compat.semantic_distinct((a,b,terminal)):continue
                neutral=[row for row in rows if row['public_tail'] and row['closed_loop']
                         and not any(r in row['role_first'] for r in (a,b,terminal))]
                if not neutral:continue
                wt=component(rows,terminal,[a,b])
                if not wt or len(wt['actions'])+1>160:continue
                for irrelevant in role_names:
                    if not compat.semantic_distinct((a,b,terminal,irrelevant)):continue
                    wi=component(rows,irrelevant,[b,terminal],closed=True,prefer_movement=True)
                    outgoing_i=component(rows,irrelevant,[b,terminal],prefer_movement=True)
                    if not outgoing_i:continue
                    if min_irrelevant_forward and (not wi or wi['forward_actions']<min_irrelevant_forward):continue
                    roles={name:compat.planner.role_spec(json.loads(sig)) for name,sig in
                        [('anchor_A',a),('anchor_B',b),('terminal',terminal),('irrelevant',irrelevant)]}
                    selected={name:rows[0]['registry'][sig] for name,sig in
                        [('anchor_A',a),('anchor_B',b),('terminal',terminal),('irrelevant',irrelevant)]}
                    tail=neutral[0]
                    closed=bool(require_closed_anchors and wi)
                    history=None;continuations=None;limits=False
                    if closed:
                        history=dict(H_A=wa['actions']+tail['actions'],H_B=wb['actions']+tail['actions'],
                            H_A_I=wa['actions']+wi['actions']+tail['actions'])
                        c0=wt['actions']+['S']
                        continuations=dict(C0=c0,C_A=wa['actions']+c0,C_B=wb['actions']+c0)
                        limits=max(map(len,history.values()))<=512 and max(map(len,continuations.values()))<=160
                    components=dict(anchor_A=wa,anchor_B=wb,terminal_only=wt,irrelevant_loop=wi,irrelevant_outgoing=outgoing_i,
                        public_tail=dict(trace_id=tail['id'],trace_ref=tail['trace_ref'],trace_digest=tail['trace_digest'],actions=tail['actions']))
                    identity=dict(house_id=rows[0]['house_id'],scene_fingerprint=rows[0]['scene_fingerprint'],hub_key=hub,roles=roles,
                        components={k:v['trace_id'] if v else None for k,v in components.items()})
                    proposal=dict(proposal_id='WB_'+compat.digest(identity)[:24],house_id=rows[0]['house_id'],
                        scene_fingerprint=rows[0]['scene_fingerprint'],hub_key=hub,hub_pose=rows[0]['hub_pose'],
                        roles=roles,eligible=selected,tasks=compat.planner.task_spec(roles),components=components,
                        histories_unbalanced=history,continuations_unreplayed=continuations,
                        history_action_counts={k:count_profile(v) for k,v in history.items()} if history else None,
                        all_components_stored=closed,unbalanced_plan_within_action_caps=limits,
                        irrelevant_has_at_least_2F=bool(wi and wi['forward_actions']>=2),
                        irrelevant_actionlist_differs_from_A_B=bool(wi and wi['actions']!=wa['actions'] and wi['actions']!=wb['actions']),
                        same_action_count_control_required=True,history_padding_and_common_join_required=True,
                        physical_replay_certified=False,training_admission=False,scientific_pass=False,
                        proof_scope='stored_component_trace_conditions_only_no_composition_proof')
                    enumerated+=1;proposals.append(proposal)
    proposals.sort(key=lambda p:(-p['all_components_stored'],-p['irrelevant_has_at_least_2F'],
        -p['irrelevant_actionlist_differs_from_A_B'],-p['unbalanced_plan_within_action_caps'],
        sum(len(c['actions']) for c in p['components'].values() if c),p['house_id'],p['proposal_id']))
    # Unordered A/B and deterministic identity avoid counting simple label swaps.
    assert len({p['proposal_id'] for p in proposals})==len(proposals)
    return dict(status='CPU_PROPOSALS_NOT_PHYSICAL_CERTIFICATES',input_traces=n,hub_groups=len(groups),
        compatible_anchor_pairs=pair_count,enumerated_programs=enumerated,proposals=proposals[:max_programs],
        truncated_programs=max(0,len(proposals)-max_programs),gpu_operations=0,new_certified_families=0,
        filters=dict(require_closed_anchors=require_closed_anchors,min_irrelevant_forward=min_irrelevant_forward))
