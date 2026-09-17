"""Closed six-house bank -> up to 24 distinct semantic programs per actual hub."""
import collections
import copy
import importlib.util
import json
from pathlib import Path
import time
import types

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('multi_bank_core',HERE/'core.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
SCOUTS=[c.WF/'scout_v1/run_v1',c.WF/'scout_next_v1/shard_0/run_v1']
SPIN_RUNS=[c.WF/'batch_execution_v1'/name/'run_v1' for name in ('batch_00','batch_01r1','batch_02r2')]
def save(path,value):
    path=Path(path);assert path.is_relative_to(HERE);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def all_bank_proposals(records):
    # Pure enumeration bound only: never change a physical/semantic threshold.
    path=c.WF/'bank_cpu/bank.py';source=path.read_text()
    old='not 1<=max_programs<=1000';assert source.count(old)==1
    source=source.replace(old,'not 1<=max_programs<=100000')
    m=types.ModuleType('multi_bank_bounded_enumeration');m.__file__=str(path)
    exec(compile(source,str(path),'exec'),m.__dict__)
    return m.propose(records,max_programs=100000)

def main():
    out=HERE/'snapshot_v3';assert not out.exists();out.mkdir()
    start=time.monotonic();lock={};cache={};trace_proofs={};pixel_cache={};spin_witnesses=[];spin_ledger=[]
    def read(path,jsonl=False):
        raw,h=c.stable(path);lock[str(path)]=h
        if jsonl:
            assert raw.endswith(b'\n'),'PARTIAL_BANK_LINE';return [json.loads(x) for x in raw.splitlines()]
        return json.loads(raw)
    for run in SPIN_RUNS:
        if not (run/'journal/HEAD.json').exists():
            spin_ledger.append({'source':str(run),'status':'NO_COMMITTED_JOURNAL'});continue
        cfg,journal=c.read_committed_prefix(run,out/'source_journal_snapshots'/run.parent.name,lock)
        byid={r['candidate_id']:r for r in cfg['candidates']};found=0
        for row in journal:
            if row['kind']!='trace_saved' or row['payload'].get('complete') is not True:continue
            payload=row['payload'];candidate=byid[payload['bundle']]
            path=run/'bundles'/payload['bundle']/'traces'/('%06d.json'%payload['index'])
            trace=read(path)
            assert lock[str(path)]==payload['sha256'],'TRACE_NOT_COMMITTED_AT_HEAD'
            actions=trace['actions']
            if actions not in (['L']*24,['R']*24):continue
            if not c.compiler.complete(trace) or not c.closed(trace['observations'][0]['pose'],trace['observations'][-1]['pose']):
                spin_ledger.append({'trace':str(path),'status':'SPIN_INCOMPLETE_OR_NOT_CLOSED'});continue
            assert trace['trace_hash']==c.digest({k:v for k,v in trace.items() if k!='trace_hash'})
            c.verify_semantic_pixels(trace,run/'content',lock,pixel_cache)
            spin_witnesses.append({'house_id':candidate['house_id'],'assets_digest':c.digest(candidate['assets']),
                'pose':trace['observations'][0]['pose'],'direction':actions[0],'trace':trace,
                'trace_ref':str(path.relative_to(c.ROOT)),'sha256':lock[str(path)],'committed_journal_seq':row['seq']})
            found+=1
        spin_ledger.append({'source':str(run),'status':'HEAD_PREFIX_VERIFIED','actual_complete_spins':found,
            'source_is_active_possible':not (run/'result.json').exists(),'journal_rows':len(journal)})
    selected=[];proposal_ledger=[];count_ledger={};group_decisions=[];hub_rows=[];spin_evidence={}
    for run in SCOUTS:
        result=read(run/'result.json');assert result['status']=='SCOUT_CLOSED' and result['error'] is None
        assert read(run/'STORE_CLOSE_AUDIT.json')['audit_pass'] is True
        cfg=read(run/'EXECUTION_CONFIG.json');head=read(run/'journal/HEAD.json')
        raw,h=c.stable(run/'journal/events.jsonl');lock[str(run/'journal/events.jsonl')]=h
        journal=c.acceptance.parse_journal(raw,head,cfg)
        committed={str(run/r['payload']['path']):r['payload']['sha256'] for r in journal if r['kind']=='trace_saved'}
        for allowed in cfg['candidates']:
            house=allowed['house_id'];folder=run/'houses'/house
            assert read(folder/'result.json')['status']=='SCOUT_COMPONENT_BANK_COMPLETE'
            records=read(folder/'BANK_RECORDS.jsonl',True);hubs=read(folder/'FROZEN_HUB_CONFIGS.json')['hubs']
            inventory=read(folder/'SEMANTIC_INVENTORY.json')['objects']
            actual=c.bank.compat.role_groups(inventory)
            for record in records:
                path=c.ROOT/record['trace_ref']
                trace=read(path);cache[str(path)]=trace
                assert lock[str(path)]==committed[str(path)] and trace==record['trace'],'BANK_TRACE_COMMIT_BINDING'
                trace_proofs[str(path)]={'complete':c.compiler.complete(trace),'digest':c.digest(trace),
                    'closed':c.closed(trace['observations'][0]['pose'],trace['observations'][-1]['pose'])}
            for hub_index,hub in enumerate(hubs):
                subset=[r for r in records if r['trace']['observations'][0]['pose']['position']==hub['position']]
                offered=all_bank_proposals(subset);viable=[];solutions_cache={}
                for proposal in offered['proposals']:
                    entry={'house_id':house,'hub_index':hub_index,'proposal_id':proposal['proposal_id'],
                           'semantic_program':c.canonical_program(proposal),'status':None}
                    proposal_ledger.append(entry)
                    if not proposal['all_components_stored']:entry['status']='MISSING_CLOSED_COMPONENT';continue
                    assert proposal['scene_fingerprint']==c.digest(allowed['assets'])
                    for role,value in proposal['roles'].items():
                        signature=(value['mpcat40'],value['room'],value['raw_match']['value'])
                        assert actual.get(signature)==proposal['eligible'][role],'SEMANTIC_INSTANCE_IDENTITY'
                    components={name:list(proposal['components'][old]['actions']) for name,old in
                        [('a','anchor_A'),('b','anchor_B'),('i','irrelevant_loop'),('terminal','terminal_only')]}
                    paths={}
                    for name,old in [('a','anchor_A'),('b','anchor_B'),('i','irrelevant_loop'),('terminal','terminal_only')]:
                        component=proposal['components'][old];path=c.ROOT/component['trace_ref'];trace=cache[str(path)]
                        proof=trace_proofs[str(path)]
                        assert proof['complete'] and proof['digest']==component['trace_digest']
                        assert trace['actions'][:len(components[name])]==components[name]
                        if name!='terminal':assert trace['actions']==components[name] and proof['closed']
                        paths[name]=trace
                    count_key=c.digest([components[x] for x in ('a','b','i')])
                    if count_key not in solutions_cache:
                        plans,attempts=c.solve(components['a'],components['b'],components['i']);solutions_cache[count_key]=plans
                        count_ledger[count_key]={'components':{x:components[x] for x in ('a','b','i')},'attempts':attempts,'solutions':plans}
                    plans=solutions_cache[count_key];entry['count_ledger_id']=count_key
                    if not plans:entry['status']='NO_K_J_COUNT_SOLUTION';continue
                    maxcont=max(len(components['a']),len(components['b']))+len(components['terminal'])+1
                    if maxcont>160:entry['status']='CONTINUATION_ACTION_CAP';continue
                    comp=c.Compiler({k:(v['mpcat40'],v['room']) for k,v in proposal['roles'].items()},proposal['tasks'],proposal['eligible'])
                    matching=[w for w in spin_witnesses if w['house_id']==house and w['assets_digest']==c.digest(allowed['assets'])
                              and c.closed(w['pose'],proposal['hub_pose'])]
                    required=sorted({d for p in plans for d in p['required_neutral_spin_directions']})
                    _,all_spin_evidence=c.spin_verdict(comp,required,matching)
                    evidence_id=c.digest(all_spin_evidence);spin_evidence.setdefault(evidence_id,all_spin_evidence)
                    acceptable=[];spin_checks=[]
                    for plan in plans:
                        evidence={d:all_spin_evidence[d] for d in plan['required_neutral_spin_directions']}
                        ok=not any(v['status']=='REJECTED_OBSERVED_ANCHOR_EVENT' for v in evidence.values())
                        spin_checks.append({'k':plan['k'],'j':plan['j'],'allowed':ok,'directions':list(evidence),
                            'spin_evidence_id':evidence_id})
                        if ok:acceptable.append((plan,evidence))
                    entry['spin_checks']=spin_checks
                    if not acceptable:entry['status']='ALL_COUNT_PLANS_REJECTED_BY_ACTUAL_DIRECTION_SPIN';continue
                    plan,spins=acceptable[0]
                    estimates={}
                    for cname,parts in [('C0',['terminal']),('C_A',['a','terminal']),('C_B',['b','terminal'])]:
                        actions=[];obs=[]
                        for part in parts:
                            n=len(components[part]);fragment=paths[part]['observations'][:n+1]
                            obs+=copy.deepcopy(fragment if not obs else fragment[1:]);actions+=components[part]
                        for i,o in enumerate(obs):o['step']=i
                        estimates[cname]=len(comp.query_from_trace({'actions':actions+['S'],'observations':obs,'complete':True,'collisions':0})['sequence'])
                    entry['source_observation_query_estimates']=estimates
                    if max(estimates.values())>160:entry['status']='STORED_OBSERVATION_QUERY_LENGTH_GT160';continue
                    ident=c.digest({'source_proposal':proposal['proposal_id'],'k':plan['k'],'j':plan['j'],'version':'multi_program_bank_v1'})[:24]
                    candidate={'candidate_id':'WF_MULTI_'+ident,'house_id':house,'hub_index':hub_index,'hub_key':proposal['hub_key'],
                        'canonical_program_id':c.semantic_id(proposal),'canonical_program':c.canonical_program(proposal),
                        'roles':proposal['roles'],'tasks':proposal['tasks'],'expected_eligible':proposal['eligible'],
                        'assets':allowed['assets'],'scene_glb':allowed['scene_glb'],'split':'FIT','components':components,
                        'balance':{'k':plan['k'],'j':plan['j']},'configuration':{'u_position':hub['position'],'yaw_bin':hub['yaw_bin'],'public_tail':hub['public_tail']},
                        'history_actions_before_public_tail':plan['history_actions'],'max_continuation_actions':maxcont,
                        'source_proposal_id':proposal['proposal_id'],'required_neutral_spin_directions':plan['required_neutral_spin_directions'],
                        'stored_spin_screening':spins,'source_observation_query_estimates':estimates,
                        'query_estimates_are_not_physical_certificates':True,'winding_padding_plan':plan['pads'],
                        'target_alignment':plan['target_alignment'],'expected_history_action_counts':plan['action_counts'],
                        'control_type':c.CONTROL_TYPE,'H_A_also_contains_I':True,'scene_group_id':'mp3d:'+house,
                        'same_hub_programs_statistically_independent':False,'runtime_allowed':False,'executable':False,'training_admission':False,
                        'component_provenance':{'source_proposal_id':proposal['proposal_id'],'canonical_program_id':c.semantic_id(proposal),
                            'selection_rule':'max24 distinct unordered A/B category-room + T category-room; min history then continuation per program; source spin direction screened',
                            'source_components':{k:{'trace_ref':v['trace_ref'],'trace_digest':v['trace_digest'],'file_sha256':lock[str(c.ROOT/v['trace_ref'])]}
                                for k,v in proposal['components'].items() if k in ('anchor_A','anchor_B','irrelevant_loop','terminal_only')},
                            'scout_configuration_sha256':lock[str(run/'EXECUTION_CONFIG.json')]}}
                    entry.update(status='COUNT_MATCHED_STORED_SPIN_SCREENED_CANDIDATE',candidate_id=candidate['candidate_id']);viable.append(candidate)
                chosen,decisions=c.select_programs(viable,24);selected.extend(chosen);group_decisions.extend(decisions)
                hub_rows.append({'house_id':house,'hub_index':hub_index,'programs_enumerated':offered['enumerated_programs'],
                    'programs_retained':len(offered['proposals']),'programs_truncated':offered['truncated_programs'],
                    'count_and_spin_viable_proposals':len(viable),'distinct_semantic_programs_selected':len(chosen)})
                print(json.dumps({'cpu_hub_complete':hub_rows[-1],'elapsed':time.monotonic()-start}),flush=True)
    selected.sort(key=lambda r:(r['house_id'],r['hub_index'],c.rank(r),r['canonical_program_id']))
    for path in [HERE/'core.py',HERE/'prepare.py',HERE/'test_core.py',c.WF/'bank_cpu/bank.py',c.WF/'winding_balance_v1/method.py',
                 c.RUNTIME/'core_bridge.py',c.RUNTIME/'loader.py',c.RUNTIME.parent/'mechanism_factory_v2/compiler.py',
                 c.WF/'quality_cpu/batch_acceptance_v1/acceptance.py']:
        _,h=c.stable(path);lock[str(path)]=h
    for path,h in lock.items():assert c.stable(Path(path))[1]==h,'LOCKED_SOURCE_CHANGED_DURING_PREPARATION'
    products={'CONFIG_DRAFT.json':{'node':'Q35N_MULTI_PROGRAM_BANK_V1','factory_variant':'winding_v1',
        'runtime_allowed':False,'executable':False,'training_allowed':False,'candidates':selected,'query_sequence_limit':160,
        'seed_replays':[1109,2209,3309],'resource_budget':None,'gpu_device':None,'requires_main_agent_admission':True},
        'SOURCE_LOCK.json':lock,'PROPOSAL_LEDGER.json':proposal_ledger,'COUNT_LEDGER.json':count_ledger,
        'SEMANTIC_DEDUP_LEDGER.json':group_decisions,'SPIN_SOURCE_LEDGER.json':spin_ledger,'SPIN_DECISION_EVIDENCE.json':spin_evidence,
        'result.json':{'status':'MULTI_PROGRAM_CANDIDATES_NOT_PHYSICAL_FAMILIES','selected_candidates':len(selected),
            'selected_houses':len({r['house_id'] for r in selected}),'hubs':hub_rows,'source_complete_spin_traces':len(spin_witnesses),
            'gpu_operations':0,'new_physical_families':0,'scientific_pass':False,'wall_seconds':time.monotonic()-start}}
    for name,value in products.items():save(out/name,value)
    print(json.dumps(products['result.json'],indent=2))
if __name__=='__main__':main()
