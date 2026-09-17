"""Freeze four new R2R-source positions in a previously rich FIT house, CPU only."""
import collections
import copy
import gzip
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent;WF=HERE.parent;ROOT=next(p for p in HERE.parents if p.name=='vla')
HOUSE='8WUmhLawc2A'
SOURCE=WF/'bulk_source_v1/next_shard_runtime_v1/run_v1'
R2R=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
selector=load('newhub_CPU_selector',HERE/'selector.py')
r=load('newhub_CPU_runtime_import',HERE/'runtime.py')
def save(path,v):
    assert path.parent==HERE
    with path.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    assert not (HERE/'SOURCE_LOCK.json').exists(),'NO_OVERWRITE'
    # Reuse actual closed-next-source verifier, never an active-source shortcut.
    p=load('newhub_closed_source_verifier',WF/'multi_program_bank_v3/prepare.py').load()
    lock=p.closure();cfg=p.read(SOURCE/'EXECUTION_CONFIG.json')
    row=copy.deepcopy(next(c for c in cfg['candidates'] if c['house_id']==HOUSE))
    positions=p.read(WF/'bulk_source_v1/SOURCE_POSITIONS.json')[HOUSE]['positions']
    assert positions==row['source_positions']
    hubpath=SOURCE/'houses'/HOUSE/'FROZEN_HUB_CONFIGS.json';old=p.read(hubpath)
    excluded=[h['position'] for h in old['hubs']];prior_refs=[str(hubpath)]
    # Include current immutable declared batch configurations conservatively;
    # no observations or outcomes of active batches are inspected.
    declared=[]
    for path in sorted((WF/'batch_execution_v1').glob('batch_*/run_v1/EXECUTION_CONFIG.json')):
        matched=[c for c in p.read(path)['candidates'] if c['house_id']==HOUSE]
        if matched:
            lock[str(path)]=p.sha(path);prior_refs.append(str(path))
            for c in matched:
                pos=c['configuration']['u_position'];excluded.append(pos)
                declared.append({'source':str(path),'candidate_id':c['candidate_id'],'position':pos,'observed_status':'CONFIG_ONLY_NO_OUTCOME_READ'})
    excluded=[list(x) for x in sorted({tuple(x) for x in excluded})]
    selected,ledger=selector.choose(positions,old['geometry']['geometry_checks'],excluded)
    assert len(selected)==4,'INSUFFICIENT_NONOVERLAPPING_PRIOR_GEOMETRY_HUBS'
    chosen=[x['position'] for x in selected]
    assert R2R in map(Path,lock) and lock[str(R2R)]==p.sha(R2R)
    with gzip.open(R2R,'rt') as f:episodes=json.load(f)['episodes']
    provenance={tuple(point):[] for point in positions}
    for e in episodes:
        if Path(e['scene_id']).parent.name!=HOUSE:continue
        start=e['start_position'];provenance[tuple(start)].append({'episode_id':e['episode_id'],'source_field':'start_position',
            'position':start,'raw_start_rotation':e['start_rotation'],'rotation_is_not_used_for_scout_reset':True})
        for i,point in enumerate(e['reference_path']):provenance[tuple(point)].append({'episode_id':e['episode_id'],
            'source_field':'reference_path','path_index':i,'position':point,'source_orientation':None})
    assert all(provenance[tuple(point)] for point in positions)
    canonical=[]
    for pos in chosen:
        canonical.append({'position':pos,'yaw_bin':0,'agent_rotation_wxyz':[1,0,0,0],
            'sensors':{name:{'position':[pos[0],pos[1]+1.25,pos[2]],'rotation_wxyz':[1,0,0,0]} for name in ('rgb','semantic')},
            'pose_status':'REQUESTED_FROM_FROZEN_BACKEND_RESET_RULE_NOT_NEW_OBSERVED_POSE',
            'reset_source':str(WF.parent/'habitat_backend.py'),'official_source_refs':provenance[tuple(pos)],
            'full_actual_agent_and_sensor_pose_must_be_saved_by_original_trace_runner':True})
    bank_result=WF/'multi_program_bank_v3/snapshot_v1/result.json';bank=p.read(bank_result)
    supply=[h for h in bank['hubs'] if h['house_id']==HOUSE];assert sum(h['distinct_semantic_programs_selected'] for h in supply)==48
    row['source_positions']=chosen
    prepared=copy.deepcopy(cfg)
    for key in ('main_agent_approval_sha256','source_lock_sha256'):prepared.pop(key,None)
    prepared.update(node='Q35N_NEW_HUB_SCOUT_FOUR_V1',candidates=[row],runtime_allowed=False,executable=False,training_allowed=False,
        runtime_adapter_ready=False,gpu_device=2,gpu_uuid=r.a.UUID,shard_id='new_hub_4_v1',
        intended_output_root=str((HERE/'run_v1').relative_to(ROOT)),
        budget={'total_actions':40000,'total_seconds':2700,'discovery_actions':40000,'discovery_seconds':2400,
                'certification_actions':1,'certification_seconds':1},
        new_hub_plan={'house_id':HOUSE,'limit':4,'selected_positions':chosen,'excluded_positions':excluded,
            'prior_hub_sources':prior_refs,'house_split':'FIT','scene_group_id':'mp3d:'+HOUSE,
            'sampling':'supply_enriched_efficiency_pilot_not_generalization_evaluation','original_event_and_component_rules_unchanged':True})
    assert row['split']=='FIT' and prepared['scene_group_policy']==cfg['scene_group_policy']
    save(HERE/'PREPARED_CONFIG.json',prepared)
    save(HERE/'HUB_SELECTION_LEDGER.json',{'house_id':HOUSE,'input_positions':len(positions),'selected_count':4,
        'selection_rule':'prior geometry reachable_groups descending, full3D position stable tie; >=1m all previous/declared/selected hubs',
        'reason_for_house':'48 distinct source-bank programs at both existing hubs; supply-only purposive efficiency sample',
        'supply':supply,'rows':ledger,'prior_declared_positions':declared,'prior_observed_hubs':old['hubs'],
        'same_house_group_preserved':True,'geometric_prior_not_new_physical_pass':True,'scientific_pass':False})
    save(HERE/'POSE_SOURCE_PROVENANCE.json',{'house_id':HOUSE,'official_source':str(R2R),'requested_hub_poses':canonical,
        'all_source_positions_with_raw_refs':[{'position':pos,'references':provenance[tuple(pos)]} for pos in positions],
        'do_not_count_rotation_variants_as_new_hubs':True,'scientific_pass':False})
    result=p.read(SOURCE/'result.json');house_result=next(h for h in result['houses'] if h['house_id']==HOUSE)
    store=p.read(SOURCE/'STORE_CLOSE_AUDIT.json')
    save(HERE/'RESOURCE_ESTIMATE.json',{'basis_closed_scout':str(SOURCE),'old_same_house_hubs':2,
        'old_same_house_actions':house_result['actual_actions'],'four_hub_linear_action_estimate':house_result['actual_actions']*2,
        'four_hub_max_component_trace_attempts':4*(1+12*2*2),
        'four_hub_worst_action_bound_before_budget_censor':4*(8+12*2*(140+504)),
        'total_motion_budget':40000,'budget_censor_is_possible_not_physical_negative':True,
        'elapsed_linear_estimate_from_six_hub_run_seconds':result['wall_seconds']*4/6,
        'content_linear_estimate_from_six_hub_run_bytes':store['committed_content_bytes']*4/6,
        'worker_total_seconds':2700,'discovery_seconds':2400,'supervisor_seconds':3000,
        'content_bytes_cap':6*1024**3,'supervisor_disk_bytes_cap':7*1024**3,'RAM_bytes_cap':8*1024**3,
        'estimates_not_guaranteed_or_new_measurements':True,'scientific_pass':False})
    for name,fn in [('COMMON_DRAFT.py',r.a.common_source),('WORKER_DRAFT.py',r.a.worker_source),('SUPERVISOR_DRAFT.py',r.a.supervisor_source)]:
        text=fn();compile(text,str(HERE/name),'exec')
        with (HERE/name).open('x') as f:f.write(text)
    for path in [hubpath,bank_result,R2R,WF/'multi_program_bank_v3/prepare.py',WF/'multi_program_bank_v2/prepare.py',
                 WF/'multi_program_bank_v1/core.py',WF/'bulk_source_v1/SOURCE_POSITIONS.json',WF.parent/'habitat_backend.py',*HERE.iterdir()]:
        assert path.is_file();lock[str(path)]=p.sha(path)
    save(HERE/'SOURCE_LOCK.json',lock)
    summary={'status':'FOUR_NEW_HUBS_CPU_PREPARED_REQUIRES_MAIN_RUNTIME_APPROVAL','house_id':HOUSE,
        'official_source_positions':len(positions),'selected_new_3D_positions':chosen,
        'exclusion_positions':len(excluded),'source_lock_entries':len(lock),
        'selection_status_counts':dict(collections.Counter(x['status'] for x in ledger)),
        'new_actual_hubs':0,'new_certified_families':0,'runtime_allowed':False,'gpu_operations':0,'scientific_pass':False}
    save(HERE/'result.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
