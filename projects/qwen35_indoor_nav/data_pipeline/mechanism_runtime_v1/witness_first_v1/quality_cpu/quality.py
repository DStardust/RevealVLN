"""Strict CPU family evidence audit; draft batch gate is never paper/model PASS."""
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
LINE=RUNTIME.parents[1]

def load(name,filename):
    path=RUNTIME/filename
    seal={r.split(None,1)[1]:r.split(None,1)[0] for r in (RUNTIME/'SHA256SUMS').read_text().splitlines()}
    if hashlib.sha256(path.read_bytes()).hexdigest()!=seal[filename]:raise ValueError('SOURCE_CHANGED')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

bridge=load('quality_core','core_bridge.py')
loader_module=load('quality_loader','loader.py')
FamilyLoader,Compiler=loader_module.FamilyLoader,bridge.Compiler

def require(value,reason):
    if not value:raise ValueError(reason)

def safe(path):
    path=Path(path).absolute()
    require(path==path.resolve() and path.is_relative_to(LINE),'NONCANONICAL_OR_OUTSIDE_LINE')
    return path

def digest(value):return bridge.digest(value)

def matrix_checks(histories,cells):
    require(set(histories)=={'H_A','H_B','H_A_I'},'HISTORY_IDS')
    require(len({tuple(v) for v in histories.values()})==3,'DUPLICATE_HISTORIES')
    require(all(set(v)<=set('FLR') for v in histories.values()),'NONMOTION_HISTORY')
    counts={h:{a:Counter(v)[a] for a in 'FLR'} for h,v in histories.items()}
    require(len({tuple(v.values()) for v in counts.values()})==1,'ACTION_COUNT_SHORTCUT')
    require(len(cells)==18,'CELL_COUNT')
    tasks=sorted({c['task_id'] for c in cells});queries=sorted({c['continuation_id'] for c in cells})
    require(len(tasks)==2 and len(queries)==3,'TASK_QUERY_DIMENSIONS')
    table={}
    for c in cells:
        key=(c['history_id'],c['task_id'],c['continuation_id'])
        require(key not in table and c['y'] in (0,1) and type(c['y']) is int,'DUPLICATE_OR_UNKNOWN_LABEL')
        table[key]=c['y']
    expected={(h,t,q) for h in histories for t in tasks for q in queries}
    require(set(table)==expected,'INCOMPLETE_CROSS_GRID')
    reversals=[(t,q) for t in tasks for q in queries if table['H_A',t,q]!=table['H_B',t,q]]
    require(bool(reversals),'NO_HISTORY_REVERSAL')
    require(all(table['H_A',t,q]==table['H_A_I',t,q] for t in tasks for q in queries),'A_CONTROL_CHANGES_LABEL')
    task_differences=[(h,q) for h in histories for q in queries if table[h,tasks[0],q]!=table[h,tasks[1],q]]
    require(bool(task_differences),'NO_TASK_DEPENDENCE')
    return {'action_counts':counts,'history_reversals':reversals,'task_differences':task_differences}


def audit_family(export_root,evidence):
    """evidence has sealed_files {absolute_path: SHA256} and explicit paths.

    Required named paths: candidate, certificate, readback, result, store_close,
    control_evidence, budget_final, resource_final; replay_paths has exactly 27
    actual full trace paths. Seals are caller-frozen inputs, NOT minted here.
    Caller must keep original failed runtime result distinct from any independently
    recomputed postprocess certificate/result; a source export error is not proof
    that physical traces are invalid. Unknown/missing evidence never passes.
    """
    report={'quality_pass':False,'grade':'UNVERIFIED','scientific_pass':False,'model_gain_pass':False,
            'control_type':None,'m2_status':'MISSING_OR_UNVERIFIED','errors':[],
            'language_scope':'controlled_task_instructions_only_no_natural_language_transfer_claim'}
    try:
        root=safe(export_root);seals=evidence.get('sealed_files',{})
        require(bool(seals),'MISSING_EXTERNAL_SEAL')
        read_total=0
        def checked(path,max_bytes=256*1024**2):
            nonlocal read_total
            path=safe(path);require(str(path) in seals,'UNSEALED_FILE:'+str(path))
            require(path.stat().st_size<=max_bytes,'INDIVIDUAL_READ_CAP')
            raw=path.read_bytes();read_total+=len(raw)
            require(read_total<=12*1024**3,'TOTAL_READ_CAP')
            require(hashlib.sha256(raw).hexdigest()==seals[str(path)],'SEALED_HASH_MISMATCH')
            return raw
        def document(name):return json.loads(checked(evidence[name]))
        manifest=json.loads(checked(root/'MANIFEST.json'))
        checked(root/'SHA256SUMS')
        export_seal_rows=(root/'SHA256SUMS').read_text().splitlines()
        require(len(export_seal_rows)<=64,'EXPORT_FILE_COUNT_CAP')
        for row in export_seal_rows:
            expected,relative=row.split('  ',1)
            require(hashlib.sha256(checked(root/relative)).hexdigest()==expected,'EXPORT_SEAL_MISMATCH')
        cfg=manifest['compiler_config'];compiler=Compiler(cfg['roles'],cfg['tasks'],cfg['eligible'],task_revision=cfg['task_revision'])
        loader=FamilyLoader(root,compiler)
        actual_readback=loader.validate_supervision_contract()
        report['m2_status']='COMPLETE_RECOMPUTED_STRONG_PROGRAM_STATE'
        candidate=document('candidate');cert=document('certificate');readback=document('readback');result=document('result')
        store=document('store_close');control=document('control_evidence');budget=document('budget_final');resource=document('resource_final')
        require(candidate==manifest['candidate'],'CANDIDATE_EXPORT_MISMATCH')
        require(candidate.get('candidate_hash')==digest({k:v for k,v in candidate.items() if k!='candidate_hash'}),'CANDIDATE_HASH')
        require(readback==actual_readback,'READBACK_NOT_RECOMPUTED_EQUIVALENT')
        require(store.get('audit_pass') is True and store.get('poisoned') is False,'STORE_NOT_CLOSED_VALID')
        require(budget.get('active') is None,'BUDGET_STILL_ACTIVE')
        bridge.BudgetLedger(budget['limits'],state=budget,clock=lambda:budget['last_clock'])
        require(budget['total_reserved_actions']<=budget['limits']['total_actions'],'ACTION_BUDGET_EXCEEDED')
        require(resource.get('own_renderer_absent') is True,'GRAPHICS_RESOURCE_FINAL_MISSING')
        require(result.get('physical_certified') is True or result.get('physical_replay_verified') is True,'NO_RECOMPUTED_PHYSICAL_ACCEPTANCE')
        require(result.get('error') is None,'POSTPROCESS_RESULT_ERROR')
        require(control.get('house_split')==evidence.get('house_split') and control.get('source_scope')==evidence.get('source_scope'),'SOURCE_SPLIT_NOT_BOUND_TO_SEALED_CONTROL')
        report['house_id']=candidate['context']['house_id'];report['hub_position']=candidate['position']
        report['control_type']=control.get('control_type')
        require(report['control_type'] in ('completed_subgoal_revisit_placement_not_event_free_detour','event_free','spatial_detour'),'UNKNOWN_CONTROL_TYPE')
        report['matrix']=matrix_checks(candidate['histories'],loader.cells)
        cutoff=len(candidate['histories']['H_A']);canonical={};queries={};short=None;policy_windows={}
        require(8<cutoff<=512 and all(0<len(v)<=160 and v[-1]=='S' for v in candidate['continuations'].values()),'DECLARED_SEQUENCE_CAP')
        for c in loader.cells:
            trace=loader.read(c['trace_path']);h,q,t=c['history_id'],c['continuation_id'],c['task_id']
            require(trace['actions']==candidate['histories'][h]+candidate['continuations'][q],'TRACE_ACTION_MISMATCH')
            require(c['prefix_cutoff']==cutoff,'PREFIX_CUTOFF_MISMATCH')
            expected_task_index=0 if compiler.tasks[t]['anchor']=='anchor_A' else 1
            require(c['outcome']==bridge.FamilyFactory.expected(expected_task_index,h,q),'ORIGINAL_MATRIX_EXPECTATION')
            events=compiler.atoms(trace['observations'])
            seen={role for event in events[:cutoff+1] for role,ids in event.items() if ids}
            required={'anchor_B'} if h=='H_B' else {'anchor_A'}|({'irrelevant'} if h=='H_A_I' else set())
            forbidden={'anchor_A'} if h=='H_B' else {'anchor_B'}
            require(required<=seen and not seen&forbidden,'HISTORY_EVENT_PATTERN')
            require(not any(events[step][r] for step in range(cutoff-7,cutoff+1) for r in ('anchor_A','anchor_B','terminal')),'PUBLIC_TAIL_EVENT')
            joins=trace.get('normalization_events',[])
            require(len(joins)==1 and joins[0]['step']==cutoff-8,'NUMERICAL_JOIN_LOCATION')
            join=joins[0];raw_pose=join['raw_record']['pose'];canonical_pose=join['canonical_record']['pose']
            for first,second in [(raw_pose,canonical_pose)]+[(raw_pose['sensors'][s],canonical_pose['sensors'][s]) for s in ('rgb','semantic')]:
                require(max(bridge.factory.pose_distance(first,second))<=1e-5,'NUMERICAL_JOIN_TOO_LARGE')
            require(trace['observations'][join['step']]['pose']==canonical_pose,'JOIN_CANONICAL_POSE')
            canonical[h,q]=trace
            query=compiler.query_from_trace(compiler.slice_continuation(trace,cutoff))
            require(queries.setdefault(q,query)==query,'QUERY_DIFFERS_ACROSS_HISTORIES')
            window=[{k:o[k] for k in ('rgb_hash','semantic_hash','pose')} for o in trace['observations'][cutoff-1:cutoff+1]]
            signature=digest(window);short=signature if short is None else short
            require(signature==short,'SHORT_WINDOW_NOT_SHARED')
            expected_prefix=compiler.policy_at(trace,t,cutoff,'audit')
            semantics=compiler.policy_semantics(expected_prefix)
            require(policy_windows.setdefault(t,semantics)==semantics,'CURRENT_POLICY_WINDOW_OR_ACTIONS_DIFFER')
            for record in loader.action_stream(c['cell_id']):
                expected=compiler.policy_at(trace,t,record['causal_cutoff_step'],record['sample_id'])
                require(record==expected,'POLICY_NOT_CAUSALLY_RECOMPUTED')
            require('query' not in expected_prefix and 'm2_program_state' not in expected_prefix,'POLICY_PRIVILEGED_FIELDS')
        for prefix_id,item in loader.prefix_index.items():
            tr=canonical[item['history_id'],next(iter(candidate['continuations']))]
            for rec in loader.prefix_records(prefix_id):
                require(rec==compiler.policy_at(tr,item['task_id'],rec['causal_cutoff_step'],rec['sample_id']),'PREFIX_POLICY_MISMATCH')
            payload=loader.policy_payload(loader.prefix_records(prefix_id)[-1])
            require(set(payload)=={'task_type','instruction','rgb','executed_actions','memory_reset'},'POLICY_PAYLOAD_BOUNDARY')
        # Match actual 27 traces by seed + exact action sequence, not filenames.
        paths=evidence['replay_paths'];require(len(paths)==len(set(paths))==27,'REPLAY_PATH_COUNT')
        actual={};rows=[]
        for path in paths:
            tr=json.loads(checked(path));seed=tr['seed']
            require(tr.get('trace_hash')==digest({k:v for k,v in tr.items() if k!='trace_hash'}),'ACTUAL_TRACE_HASH')
            require(seed in (1109,2209,3309) and compiler.complete(tr),'REPLAY_SEED_OR_COMPLETENESS')
            pairs=[key for key,base in canonical.items() if tr['actions']==base['actions']]
            require(len(pairs)==1,'AMBIGUOUS_REPLAY_ACTION_IDENTITY')
            h,q=pairs[0];key=(seed,h,q);require(key not in actual,'DUPLICATE_REPLAY_CELL')
            content=digest({'actions':tr['actions'],'observations':tr['observations']})
            require(content==digest({'actions':canonical[h,q]['actions'],'observations':canonical[h,q]['observations']}),'SEED_TRACE_CONTENT_MISMATCH')
            actual[key]=content
            for task in compiler.tasks:rows.append({'task':task,'history':h,'continuation':q,'outcome':compiler.evaluate(tr,task),'seed':seed})
        require(len(actual)==27 and len(rows)==54 and cert.get('replays')==27 and cert.get('evaluations')==54,'CERTIFICATE_DIMENSIONS')
        claimed=[row for entry in cert['seeds'] for row in entry['rows']]
        require(sorted(map(digest,rows))==sorted(map(digest,claimed)),'CERTIFICATE_ROWS_NOT_RECOMPUTED')
        require(len(cert['seeds'])==3 and all(x.get('replays')==9 for x in cert['seeds']),'CERTIFICATE_SEED_GRIDS')
        # Verify every indexed content blob including semantic arrays, once.
        semantic_counts={}
        for key,item in loader.contents.items():
            raw=checked(LINE/item['line_relative_path'])
            require(hashlib.sha256(raw).hexdigest()==item['file_sha256'],'CONTENT_INDEX_HASH')
            pixels=loader_module.npy_pixels(raw,item['raw_pixel_sha256'],item['kind'])
            if item['kind']=='semantic':semantic_counts[item['raw_pixel_sha256']]=dict(Counter(str(x[0]) for x in struct.iter_unpack('<I',pixels)))
        for trace in canonical.values():
            observations=list(trace['observations'])
            for event in trace.get('normalization_events',[]):observations.extend([event['raw_record'],event['canonical_record']])
            for obs in observations:
                require({str(k):v for k,v in obs['pixels'].items() if v}==semantic_counts[obs['semantic_hash']],'SEMANTIC_PIXELS_NOT_RECOMPUTED')
        control_type=report['control_type']
        if control_type=='completed_subgoal_revisit_placement_not_event_free_detour':
            require(report['matrix']['action_counts']['H_A']['F']>=2,'REVISIT_NOT_SPATIAL')
            for h in ('H_A','H_A_I'):
                tr=canonical[h,next(iter(candidate['continuations']))]
                active=[bool(e['anchor_A']) for e in compiler.atoms(tr['observations'][:cutoff+1])]
                episodes=sum(on and (i==0 or not active[i-1]) for i,on in enumerate(active))
                require(episodes>=2,'REVISIT_REPEATED_ANCHOR_EVENT_NOT_OBSERVED')
        else:
            # A declaration alone cannot certify an event-free/spatial segment.
            ranges=control.get('intervention_ranges',{})
            require(set(ranges)=={'H_A','H_A_I'},'CONTROL_SEGMENT_EVIDENCE_MISSING')
            for h,(start,end) in ranges.items():
                require(type(start) is int and type(end) is int and 0<=start<end<=cutoff,'CONTROL_SEGMENT_BOUNDS')
                tr=canonical[h,next(iter(candidate['continuations']))];events=compiler.atoms(tr['observations'])
                protected_roles=tuple(compiler.roles) if control_type=='event_free' else ('anchor_A','anchor_B','terminal')
                require(not any(events[t][r] for t in range(start,end+1) for r in protected_roles),'CONTROL_SEGMENT_CONTAINS_TASK_EVENT')
                if control_type=='spatial_detour':require('F' in tr['actions'][start:end],'DETOUR_NOT_SPATIAL')
        report.update(cells_verified=18,replays_verified=27,evaluations_verified=54,
                      evidence_verified=True,read_bytes_counted=read_total,policy_future_query_excluded=True,
                      source_error_is_distinct_from_postprocess_acceptance=True)
        if manifest['split']=='interface_only' or evidence.get('source_scope')!='actual_physical_fit':
            report['grade']='INTERFACE_ONLY_NOT_TRAINING';return report
        require(manifest['split']=='candidate_fit_pool' and evidence.get('house_split')=='FIT','FIT_SPLIT_MISSING')
        report.update(quality_pass=True,grade='QUALITY_VERIFIED_CANDIDATE_FIT_NOT_MODEL_GAIN')
    except (KeyError,ValueError,OSError,TypeError,StopIteration) as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        report['grade']='MISSING_OR_REJECTED_EVIDENCE'
    return report


def evaluate_batch_draft(batches,families):
    """Pure draft evaluator; caller supplies separately verified reports.

    Always engineering_pass=false until main agent freezes a versioned threshold
    and a production wrapper independently binds batch seals, attempts and reports.
    """
    errors=[];qualified=[];types=Counter()
    if len(batches)!=2:errors.append('EXACTLY_TWO_PREDECLARED_BATCHES_REQUIRED')
    for batch in batches:
        attempted=batch.get('attempts',[]);frozen=batch.get('frozen_attempt_ids',[])
        if len(attempted)<3 or len(attempted)!=len(frozen) or len(frozen)!=len(set(frozen)) or set(frozen)!={r['id'] for r in attempted}:
            errors.append('ATTEMPT_DENOMINATOR_OR_FROZEN_SET')
        if not batch.get('sealed_input_verified') or not attempted or batch.get('freeze_unix',float('inf'))>=min(r['started_unix'] for r in attempted):
            errors.append('BATCH_NOT_VERIFIED_FROZEN_BEFORE_ATTEMPTS')
        accepted=[]
        for attempt in attempted:
            f=families.get(attempt['id'],{})
            if f.get('quality_pass') is True and f.get('evidence_verified') is True:
                accepted.append(f);types[f['control_type']]+=1
        if len(accepted)<2:errors.append('BATCH_FEWER_THAN_TWO_QUALIFIED')
        qualified.extend(accepted)
    # Connected-components merge same-house hubs within 0.5 m, ignoring role/yaw/
    # seed changes. This radius is an explicit DRAFT, not an approved new protocol.
    groups=[]
    for f in qualified:
        if len(f.get('hub_position',[]))!=3 or not all(math.isfinite(x) for x in f['hub_position']):
            errors.append('INVALID_HUB');continue
        touching=[g for g in groups if any(x['house_id']==f['house_id'] and math.dist(x['hub_position'],f['hub_position'])<=.5 for x in g)]
        merged=[f]
        for group in touching:merged.extend(group);groups.remove(group)
        groups.append(merged)
    houses={f['house_id'] for f in qualified}
    if len(groups)<6:errors.append('FEWER_THAN_SIX_DISTINCT_PHYSICAL_HUBS')
    if len(houses)<3:errors.append('FEWER_THAN_THREE_FIT_HOUSES')
    return {'draft_threshold_met':not errors,'engineering_pass':False,'main_agent_freeze_required':True,
            'quality_report_bindings_independently_verified_here':False,'errors':errors,
            'qualified_reports':len(qualified),'distinct_physical_hubs':len(groups),'houses':len(houses),
            'control_type_counts':dict(types),'mixed_control_types_pooled_as_same_mechanism':False,
            'statistical_generalization_pass':False,'model_gain_pass':False,'scientific_pass':False}
