"""Read-only scientific accounting followed by local evidence outputs."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]

def read(name):return json.loads((OUT/name).read_text())
def rows(name):
    p=OUT/name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def main():
    assert not (OUT/'result.json').exists()
    terminal=read('discovery_pruned_RESULT.json')
    assert terminal['decision']=='DISCOVERY_ATTEMPTS_EXHAUSTED', 'Review unexpected outcome rather than forcing a failure report'
    attempts=rows('DISCOVERY_ATTEMPTS.jsonl')
    assert len(attempts)==256 and [r['attempt'] for r in attempts]==list(range(1,257))
    assert not (OUT/'FROZEN_CANDIDATE.json').exists()
    traces={r['trace_hash']:r for r in rows('PREVIEW_TRACES.jsonl')}
    views=read('WITNESS_POOL.json'); events={}
    for k,vs in views.items():
        histogram=collections.Counter()
        for v in vs:
            tr=traces[v['trace_hash']]
            assert tr['complete'] and any(v['instance'] in o['see2'][k] for o in tr['observations'])
            pattern=','.join(sorted({kind for o in tr['observations'] for kind,ids in o['see2'].items() if ids}))
            histogram[pattern]+=1
        events[k]=dict(histogram)
    plans=rows('ROUTE_PLANS.jsonl')
    sink_plans=[p for p in plans if p['view']['kind']=='K']
    successful_sink_plans=[p for p in sink_plans if p['trace'] is not None and p['trace']['complete']]
    contaminated=[p for p in successful_sink_plans if any(o['see2']['D'] for o in p['trace']['observations'])]
    # Counterexamples are from actual stored traces, not expected paper labels.
    examples=[]
    for p in contaminated[:3]:
        tr=p['trace']; t=next(i for i,o in enumerate(tr['observations']) if o['see2']['D'])
        o=tr['observations'][t]; idx=o['see2']['D'][0]; prev=tr['observations'][t-1]
        counts=[prev['pixels'].get(str(idx),prev['pixels'].get(idx,0)),o['pixels'].get(str(idx),o['pixels'].get(idx,0))]
        assert min(counts)>=256
        examples.append({'route_key':p['key'],'trace_hash':tr['trace_hash'],'decision_step':t,
            'chair_instance':idx,'pixel_counts':counts,'rgb_hashes':[prev['rgb_hash'],o['rgb_hash']],
            'semantic_hashes':[prev['semantic_hash'],o['semantic_hash']]})
    source_proof={}
    lock=read('discovery_CODE_LOCK.json')
    for name,snapshot in [('engine.py','engine_discovery_original.py.snapshot'),('run_phase.py','run_phase_original.py.snapshot')]:
        h=hashlib.sha256((OUT/snapshot).read_bytes()).hexdigest()
        source_proof[name]={'snapshot':snapshot,'sha256':h,'matches_phase_lock':h==lock[name]}
        assert h==lock[name]
    old=(OUT/'engine_discovery_original.py.snapshot').read_text()
    preview=old.split('    def complete_candidate(self,u,bin,base,tail):')[0]+"    def complete_candidate(self,u,bin,base,tail):\n        raise Reject('CONTINUATION_STAGE_NOT_IMPLEMENTED')\n"
    source_proof['preview_engine_reconstruction']={'rule':'original discovery snapshot prefix before complete_candidate plus the original unexecuted stub',
        'computed_sha256':hashlib.sha256(preview.encode()).hexdigest(),
        'matches_preview_lock':hashlib.sha256(preview.encode()).hexdigest()==read('preview_CODE_LOCK.json')['engine.py']}
    save('SOURCE_VERSION_CHECK.json',source_proof)
    phase_counts={p:read(f'{p}_COUNTS.json') for p in ('preview','discovery','discovery_pruned')}
    totals=collections.Counter()
    for c in phase_counts.values():
        for k,v in c.items():
            if k not in ('family_candidates','family_candidate_evaluations'):totals[k]+=v
    execution={p:read(f'{p}_EXECUTION.json') for p in phase_counts}
    assert all(x['cleanup_complete'] for x in execution.values())
    resources={'phase_execution':execution,'disk_bytes':int(subprocess.check_output(['du','-sb',str(OUT)],text=True).split()[0]),
               'network_download_bytes':0,'installations':0}
    assert resources['disk_bytes']<5*1024**3
    protected={}
    for name in ('Q35N_G0R_DEPENDENCY_RECOVERY_V1','Q35N_P2R1_SPEC_CORRECTIONS_V1','Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1'):
        c=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT.parent/name,capture_output=True,text=True)
        protected[name]={'pass':c.returncode==0,'entries':c.stdout.count(': OK')}
        assert c.returncode==0
    save('PROTECTED_ARTIFACTS.json',protected)
    save('FAILURE_ANALYSIS.json',{'witness_event_patterns':events,'sink_route_plans':len(sink_plans),
        'complete_sink_route_plans':len(successful_sink_plans),'complete_sink_routes_with_chair_event':len(contaminated),
        'causal_counterexamples':examples,'family_rejections':dict(collections.Counter(x['reason'] for x in attempts)),
        'tested_distinct_positions':len({tuple(x['u']['position']) for x in attempts}),
        'tested_u_yaw_bases':len({x['base_cache_key'] for x in attempts}),
        'pool_positions':len(read('U_POOL.json')),
        'interpretation':'Endpoint witnesses are separable; this bounded greedy transit/return construction did not isolate the kitchen-only history. No impossibility or model-performance conclusion.'})
    save('RESOURCE_MEASUREMENTS.json',resources)
    save('ACTUAL_RUNTIME_COUNTS.json',{'by_phase':phase_counts,'sum_non_cumulative_counters':dict(totals),
        'unique_completed_family_configurations':256,'incomplete_candidate_rerun_for_equivalent_pruning':1,
        'note':'sim.step primitives are distinct from internal hypothetical greedy-planner actuation; both are not navigation evaluation episodes'})
    save('result.json',{'node':'Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2',
        'decision':'DISCOVERY_FAIL_UNDER_FROZEN_CONSTRUCTOR','data_contract_pass':False,
        'semantic_witness_preview_pass':True,'witness_pool_counts':{k:len(v) for k,v in views.items()},
        'candidate_configurations_completed':256,'distinct_u_yaw_bases':32,'distinct_positions':2,
        'frozen_families':0,'certification_physical_replays':0,'certification_task_evaluations':0,
        'exported_training_labels':0,'new_model_loads':0,'new_training_runs':0,'navigation_evaluation_episodes':0,
        'scientific_pass':False,'navigation_gain':None,'main_algorithm_refuted':False,
        'unreached_gates':['u/s exact physical and pixel merge','three continuations and 18-cell matrix',
            '27 certification replays and 54 task evaluations','full family schema/X01-X13 acceptance','Qwen interface and learning'],
        'runtime_unchanged':True,'gpu_cleanup_complete':True,'other_processes_stopped':0,
        'next_recommendation':'Versioned candidate-coverage correction first: visit all 32 u positions before spending budget on yaw/tail variants; no change to task/threshold/mainline',
        'next_execution_approved':False})
    print(json.dumps(read('result.json'),ensure_ascii=False))

if __name__=='__main__':main()
