"""Frozen completed-house bank -> bounded count-matched runtime drafts."""
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
ROOT=RUNTIME.parents[3]
sys.path.insert(0,str(HERE))
from method import solve,matched,closed,FamilyFactory,CONTROL_TYPE
sys.path.insert(0,str(RUNTIME))
from core_bridge import Compiler,compiler,digest
spec=importlib.util.spec_from_file_location('batch_plan_frozen_bank',WF/'bank_cpu/bank.py')
bank=importlib.util.module_from_spec(spec);spec.loader.exec_module(bank)


def stable(path):
    path=path.resolve(strict=True)
    if not path.is_relative_to(ROOT):raise ValueError('SOURCE_OUT_OF_SCOPE')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('ACTIVE_SOURCE_CHANGED')
    return raw,hashlib.sha256(raw).hexdigest()


def snapshot(output):
    output=output.resolve()
    if output.exists() or not output.is_relative_to(HERE) or output==HERE:raise ValueError('FRESH_SNAPSHOT_REQUIRED')
    source=WF/'scout_v1/run_v1';locks={};cache={};trace_digests={}
    def read(path,jsonl=False):
        raw,h=stable(path);locks[str(path.resolve())]=h
        if jsonl:
            if not raw.endswith(b'\n'):raise ValueError('PARTIAL_BANK_LINE')
            return [json.loads(line) for line in raw.splitlines()]
        return json.loads(raw)
    cfg=read(source/'EXECUTION_CONFIG.json')
    split=read(RUNTIME.parents[1]/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json')
    decisions=[];runtime=[];hubs=[];deferred=[];proposal_catalog=[]
    for allowed in cfg['candidates']:
        house=allowed['house_id'];folder=source/'houses'/house
        if allowed['split']!='FIT' or house not in split['FIT']:raise ValueError('HOUSE_NOT_FIT')
        if not (folder/'result.json').is_file():
            deferred.append({'house_id':house,'status':'HOUSE_STILL_RUNNING_NO_BANK_READ'});continue
        terminal=read(folder/'result.json')
        if terminal['status']!='SCOUT_COMPONENT_BANK_COMPLETE':
            deferred.append({'house_id':house,'status':terminal['status']});continue
        records=read(folder/'BANK_RECORDS.jsonl',True)
        frozen=read(folder/'FROZEN_HUB_CONFIGS.json')
        inventory=read(folder/'SEMANTIC_INVENTORY.json')['objects']
        normalized,_=bank.normalize_bank(records)
        for index,hub in enumerate(frozen['hubs']):
            subset=[r for r in records if r['trace']['observations'][0]['pose']['position']==hub['position']]
            if not subset:raise ValueError('FROZEN_HUB_WITHOUT_RECORDS')
            offered=bank.propose(subset,max_programs=1000)
            proposal_catalog.append({'house_id':house,'hub_index':index,**offered})
            viable=[]
            for proposal in offered['proposals']:
                decision={'house_id':house,'hub_index':index,'proposal_id':proposal['proposal_id'],
                          'candidate_status':None,'attempts':[],'accepted_count_solutions':[]}
                decisions.append(decision)
                if not proposal['all_components_stored']:
                    decision['candidate_status']='MISSING_CLOSED_COMPONENT';continue
                if proposal['scene_fingerprint']!=digest(allowed['assets']):raise ValueError('SCENE_FINGERPRINT_MISMATCH')
                if proposal['hub_pose']['position']!=hub['position']:raise ValueError('HUB_POSITION_MISMATCH')
                # Recheck roles against frozen whole-house runtime metadata.
                actual=bank.compat.role_groups(inventory)
                for role,value in proposal['roles'].items():
                    signature=(value['mpcat40'],value['room'],value['raw_match']['value'])
                    if actual.get(signature)!=proposal['eligible'][role]:raise ValueError('ROLE_INVENTORY_MISMATCH')
                components={name:list(proposal['components'][old]['actions']) for name,old in
                            [('a','anchor_A'),('b','anchor_B'),('i','irrelevant_loop'),('terminal','terminal_only')]}
                paths={}
                for name,old in [('a','anchor_A'),('b','anchor_B'),('i','irrelevant_loop'),('terminal','terminal_only')]:
                    component=proposal['components'][old];path=ROOT/component['trace_ref']
                    if str(path) not in cache:
                        cache[str(path)]=read(path)
                        if not compiler.complete(cache[str(path)]):raise ValueError('COMPONENT_TRACE_INCOMPLETE')
                        trace_digests[str(path)]=digest(cache[str(path)])
                    trace=cache[str(path)]
                    if trace_digests[str(path)]!=component['trace_digest']:raise ValueError('COMPONENT_TRACE_MISMATCH')
                    if trace['actions'][:len(components[name])]!=components[name]:raise ValueError('COMPONENT_ACTION_MISMATCH')
                    if name!='terminal' and (trace['actions']!=components[name] or not closed(trace['observations'][0]['pose'],trace['observations'][-1]['pose'])):
                        raise ValueError('COMPONENT_NOT_COMPLETE_CLOSED_LOOP')
                    paths[name]=trace
                plans,attempts=solve(components['a'],components['b'],components['i'])
                decision.update(attempts=attempts,accepted_count_solutions=plans)
                max_cont=max(len(components['a']),len(components['b']))+len(components['terminal'])+1
                if not plans:decision['candidate_status']='NO_K_J_COUNT_SOLUTION';continue
                if max_cont>160:decision['candidate_status']='CONTINUATION_ACTION_CAP';continue
                comp=Compiler({k:(v['mpcat40'],v['room']) for k,v in proposal['roles'].items()},proposal['tasks'],proposal['eligible'])
                estimates={}
                # These are recomposed STORED observations, diagnostic only.
                # Never export them as new rollout data or use them as certificates.
                for cname,parts in [('C0',['terminal']),('C_A',['a','terminal']),('C_B',['b','terminal'])]:
                    actions=[];observations=[]
                    for part in parts:
                        n=len(components[part]);fragment=paths[part]['observations'][:n+1]
                        observations+=copy.deepcopy(fragment if not observations else fragment[1:])
                        actions+=components[part]
                    for t,o in enumerate(observations):o['step']=t
                    synthetic={'actions':actions+['S'],'observations':observations,'complete':True,'collisions':0}
                    estimates[cname]=len(comp.query_from_trace(synthetic)['sequence'])
                plan=plans[0]
                ident=digest({'proposal':proposal['proposal_id'],'k':plan['k'],'j':plan['j']})[:24]
                candidate={'candidate_id':'WF_WIND_'+ident,'house_id':house,'hub_index':index,'hub_key':proposal['hub_key'],
                    'roles':proposal['roles'],'tasks':proposal['tasks'],'expected_eligible':proposal['eligible'],
                    'assets':allowed['assets'],'scene_glb':allowed['scene_glb'],'split':'FIT','components':components,
                    'balance':{'k':plan['k'],'j':plan['j']},
                    'configuration':{'u_position':hub['position'],'yaw_bin':hub['yaw_bin'],'public_tail':hub['public_tail']},
                    'winding_padding_plan':plan['pads'],'target_alignment':plan['target_alignment'],
                    'required_neutral_spin_directions':plan['required_neutral_spin_directions'],
                    'actual_spin_neutrality_pass':None,'expected_history_action_counts':plan['action_counts'],'history_actions_before_public_tail':plan['history_actions'],
                    'max_continuation_actions':max_cont,'source_observation_query_estimates':estimates,
                    'query_estimates_are_not_physical_certificates':True,'source_proposal_id':proposal['proposal_id'],
                    'control_type':CONTROL_TYPE,
                    'component_provenance':{'source_proposal_id':proposal['proposal_id'],
                        'source_hub_key':proposal['hub_key'],'source_house_id':house,
                        'selection_rule':'winding_v1_per_completed_hub_min_history_actions_then_max_continuation_actions_then_proposal_id; k2..16_j1..16; i>=2F; real_hub_spin_neutrality_required',
                        'source_components':{name:{'trace_ref':value['trace_ref'],'trace_digest':value['trace_digest'],
                            'file_sha256':locks[str((ROOT/value['trace_ref']).resolve())]} for name,value in
                            proposal['components'].items() if name in ('anchor_A','anchor_B','irrelevant_loop','terminal_only')},
                        'scout_configuration_sha256':locks[str((source/'EXECUTION_CONFIG.json').resolve())]},
                    'H_A_also_contains_I':True,'runtime_allowed':False,'executable':False,'training_admission':False}
                decision.update(candidate_status='COUNT_MATCHED_RUNTIME_CANDIDATE',candidate_id=candidate['candidate_id'],
                                source_observation_query_estimates=estimates)
                viable.append(candidate)
            viable.sort(key=lambda r:(r['history_actions_before_public_tail'],r['max_continuation_actions'],r['source_proposal_id']))
            if viable:runtime.append(viable[0])
            hubs.append({'house_id':house,'hub_index':index,'input_records':len(subset),
                         'programs_enumerated':offered['enumerated_programs'],'programs_retained':len(offered['proposals']),
                         'programs_truncated':offered['truncated_programs'],'count_matched_candidates':len(viable),
                         'selected_candidate_id':viable[0]['candidate_id'] if viable else None})
    # First round: one best hub per house, before a second same-house hub.
    by_house=collections.defaultdict(list)
    for candidate in runtime:by_house[candidate['house_id']].append(candidate)
    for rows in by_house.values():
        rows.sort(key=lambda r:(r['history_actions_before_public_tail'],r['max_continuation_actions'],r['source_proposal_id']))
        for rank,row in enumerate(rows):row['house_selection_rank']=rank
    runtime.sort(key=lambda r:(r['house_selection_rank'],r['house_id']))
    for path in [HERE/'method.py',HERE/'prepare.py',WF/'bank_cpu/bank.py',WF/'compatibility_cpu/audit.py',
                 RUNTIME.parent/'mechanism_factory_v2/factory.py',RUNTIME.parent/'mechanism_factory_v2/compiler.py']:
        _,h=stable(path);locks[str(path)]=h
    # Verify snapshot remained stable through enumeration; no mutable source copied silently.
    for path,h in locks.items():
        if stable(Path(path))[1]!=h:raise ValueError('SOURCE_CHANGED_DURING_SNAPSHOT')
    output.mkdir()
    result={'status':'COUNT_MATCHED_BATCH_DRAFT_NOT_EXECUTABLE','selected_candidates':len(runtime),
            'selected_houses':len({r['house_id'] for r in runtime}),'target_house_hubs':6,
            'selected_hubs_are_not_independent_houses':True,'hubs':hubs,'deferred_houses':deferred,
            'new_physical_families':0,'runtime_executed':False,'scientific_pass':False}
    products={'CONFIG_DRAFT.json':{'node':'Q35N_WITNESS_WINDING_BALANCE_V1','factory_variant':'winding_v1','runtime_allowed':False,'executable':False,
                    'training_allowed':False,'candidates':runtime,'query_sequence_limit':160,'seed_replays':[1109,2209,3309],
                    'resource_budget':None,'gpu_device':None,'requires_main_agent_admission':True},
              'SOURCE_LOCK.json':locks,'DECISION_LEDGER.json':decisions,'PROPOSAL_CATALOG.json':proposal_catalog,'result.json':result}
    for name,value in products.items():
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    snapshot(parser.parse_args().output)
