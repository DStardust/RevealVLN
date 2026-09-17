"""CPU arithmetic from closed runtime evidence; no yield or independence claim."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parents[1]
SCOUTS=[WF/'scout_v1/run_v1',WF/'scout_next_v1/shard_0/run_v1']
V3=WF/'short_revisit_v3/run_v1'
def read(path):return json.loads(path.read_text())
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    scouts=[read(p/'result.json') for p in SCOUTS]
    result=read(V3/'result.json');budget=read(V3/'BUDGET_FINAL.json');store=read(V3/'STORE_CLOSE_AUDIT.json')
    readback=read(V3/'bundles/WF_SHORT_REVISIT_V3_004/READBACK.json')
    quality_path=WF/'quality_cpu/batch_acceptance_v1/v3_final_audit/FAMILY_REPORT.json'
    assert read(quality_path)['quality_pass'] is True
    assert all(x['status']=='SCOUT_CLOSED' and x['error'] is None for x in scouts)
    cert=budget['bundles']['WF_SHORT_REVISIT_V3_004']['certification']
    cert_seconds=cert['finished']-cert['started']
    scout_seconds=sum(x['wall_seconds'] for x in scouts);scout_houses=sum(len(x['houses']) for x in scouts)
    # du -sb measured read-only on these closed source folders in this task.
    disk={'v3_run_apparent_bytes':141590487,'scout_run_apparent_bytes':[2076002590,2549765037],
          'measurement':'read-only du -sb; closed run trees; directory entry sizes included'}
    scenarios=[]
    for families in (1000,10000):
        scenarios.append({'target_quality_families':families,'cross_cells':families*18,
            'mandatory_actual_replays':families*27,'single_positive_certification_actions_scaled':families*cert['actions'],
            'certification_only_gpu_hours_if_same_cost_all_success':families*cert_seconds/3600,
            'full_construction_gpu_hours_if_same_cost_all_success':families*result['wall_seconds']/3600,
            'raw_run_TiB_if_same_size_no_interfamily_dedup':families*disk['v3_run_apparent_bytes']/1024**4,
            'content_TiB_if_same_size_no_interfamily_dedup':families*store['actual_disk_bytes']/1024**4,
            'CE_records_if_same_density_not_forecast':families*readback['ce_owners_verified'],
            'attempt_yield_sensitivity_not_measured':[{'assumed_yield':p,
                'gpu_hours_if_each_attempt_costs_like_positive':families*result['wall_seconds']/3600/p}
                for p in (1,.5,.2)]})
    evidence={'scout_completed_houses':scout_houses,'scout_total_seconds':scout_seconds,
        'scout_seconds_per_house':scout_seconds/scout_houses,'scout_actual_actions':sum(x['counts']['actual_actions'] for x in scouts),
        'positive_family_whole_run_seconds':result['wall_seconds'],'positive_certification_seconds':cert_seconds,
        'positive_certification_actual_actions':cert['actions'],'positive_readback_CE_unique_owners':readback['ce_owners_verified'],
        'positive_content_bytes':store['actual_disk_bytes'],**disk}
    capacity={'mechanism_FIT_houses':43,'completed_scout_houses':6,'remaining_houses':37,
        'current_max_hubs_per_house':2,'current_max_independent_hubs':86,
        'current_one_candidate_per_hub_recipe_max_selected_families':86,
        'currently_prepared_first_slice_max_hubs':6,
        '10000_families_required_mean_per_hub_if_all_86_hubs_usable':10000/86,
        'role_seed_language_variants_are_new_independent_hubs':False,
        'remaining37_scout_gpu_hours_at_observed_mean':37*scout_seconds/scout_houses/3600,
        'next18_scout_gpu_hours_at_observed_mean':18*scout_seconds/scout_houses/3600,
        'remaining37_scout_nominal_factory_budget_hours_13_shards':13*4200/3600,
        'remaining37_scout_apparent_GiB_at_observed_mean':37*sum(disk['scout_run_apparent_bytes'])/6/1024**3,
        'yield_of_quality_families_across_new_houses':None,
        'bottleneck':'current two-hub one-candidate recipe cannot reach 10000 families; discovery/certification expansion requires a separately frozen recipe and actual yield evidence'}
    save('result.json',{'status':'MEASURED_COST_SCENARIOS_NOT_PRODUCTION_FORECAST','evidence':evidence,
        'capacity':capacity,'scenarios':scenarios,'scientific_pass':False,
        'limits':['One accepted physical family is not a throughput/yield distribution.',
                  'Scenarios do not include failed-attempt overhead, training, asset downloads, or new engineering.',
                  'More distinct semantic programs at the same hub can be training samples but remain correlated.',
                  'No thresholds, replay counts, current runtime files, or frozen selection were changed.']})
    paths=[*(p/'result.json' for p in SCOUTS),V3/'result.json',V3/'BUDGET_FINAL.json',V3/'STORE_CLOSE_AUDIT.json',
           V3/'bundles/WF_SHORT_REVISIT_V3_004/READBACK.json',quality_path]
    save('SOURCE_LOCK.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps({'cert_seconds':cert_seconds,'one_family_total_seconds':result['wall_seconds'],
        'current_max_family_candidates':86,'10k_positive_cost_GPU_hours':10000*result['wall_seconds']/3600},indent=2))
if __name__=='__main__':main()
