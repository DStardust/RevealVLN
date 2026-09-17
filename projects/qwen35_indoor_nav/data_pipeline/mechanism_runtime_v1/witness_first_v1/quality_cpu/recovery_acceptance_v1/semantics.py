"""Explicit semantic-only extraction of sealed quality.py; no runtime gate simulation.

The copied semantic clauses remain inspectable source. Runtime, budget and store
closure are handled separately by recovery.py, not bypassed with fake documents.
Only completed-subgoal-revisit controls are admitted in this recovery version.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

SOURCE = Path(__file__).resolve().parent.parent/'quality.py'
EXPECTED_SOURCE = 'fdebf7078c59dda3217587e6b3ad580cf6df75dede93369700e1b79329df0e76'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=EXPECTED_SOURCE:
    raise ValueError('SEALED_SEMANTIC_SOURCE_CHANGED')
spec=importlib.util.spec_from_file_location('recovery_original_quality',SOURCE)
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
require,digest,Compiler,FamilyLoader=quality.require,quality.digest,quality.Compiler,quality.FamilyLoader
bridge,loader_module,LINE=quality.bridge,quality.loader_module,quality.LINE
matrix_checks=quality.matrix_checks

def audit_semantics(root,candidate,cert,readback,replay_paths,control_type,checked):
    report={'semantic_content_pass':False,'original_quality_pass':False,'m2_status':'MISSING_OR_UNVERIFIED'}
    read_total=0  # Reader separately inventories all actual bytes.
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
    require(candidate==manifest['candidate'],'CANDIDATE_EXPORT_MISMATCH')
    require(candidate.get('candidate_hash')==digest({k:v for k,v in candidate.items() if k!='candidate_hash'}),'CANDIDATE_HASH')
    require(readback==actual_readback,'READBACK_NOT_RECOMPUTED_EQUIVALENT')
    require(manifest['split']=='candidate_fit_pool','RECOVERY_FIT_ONLY')
    require(control_type=='completed_subgoal_revisit_placement_not_event_free_detour','RECOVERY_CONTROL_SCOPE')
    report['house_id']=candidate['context']['house_id'];report['hub_position']=candidate['position']
    report['control_type']=control_type
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
    paths=replay_paths;require(len(paths)==len(set(paths))==27,'REPLAY_PATH_COUNT')
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
        ranges={}
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
    report.update(semantic_content_pass=True,original_quality_pass=False,grade='SEMANTIC_CONTENT_VERIFIED_WITHOUT_RUNTIME_TERMINAL_ADMISSION')
    return report

