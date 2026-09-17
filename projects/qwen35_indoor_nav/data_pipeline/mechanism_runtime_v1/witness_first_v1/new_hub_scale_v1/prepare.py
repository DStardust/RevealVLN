"""CPU-only source-position coverage across legal FIT houses; no new physical claims."""
import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;WF=HERE.parent
ROOT=next(p for p in HERE.parents if p.name=='vla');LINE=ROOT/'projects/qwen35_indoor_nav'
MANIFEST=LINE/'data_pipeline/mechanism_scale_v1/acceptance_v1/candidates.json'
SPLIT=LINE/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
R2R=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
BASE=WF/'new_hub_scout_v1/PREPARED_CONFIG.json'
CLOSED=[WF/'scout_v1/run_v1',WF/'scout_next_v1/shard_0/run_v1',WF/'bulk_source_v1/shard_00/run_v1',
    WF/'bulk_source_v1/next_shard_runtime_v1/run_v1',WF/'new_hub_scout_v1/run_v1']
def read(path):return json.loads(path.read_text())
def sha(path):
    assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024**2),b''):h.update(x)
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return h.hexdigest()
def save(path,value):
    assert path.is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def point(x):
    assert len(x)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in x),'FINITE_3D_POINT'
    return tuple(x)
def select(positions,excluded,limit=4):
    assert limit==4,'FOUR_HUB_SCOUT_BUDGET_UNCHANGED'
    excluded=[point(x) for x in excluded];seen=set();eligible=[];ledger=[]
    for index,pos in enumerate(positions):
        pos=point(pos);row={'source_index':index,'position':list(pos)};ledger.append(row)
        if pos in seen:row['status']='DUPLICATE_OFFICIAL_POSITION';continue
        seen.add(pos)
        if any(math.dist(pos,x)<1 for x in excluded):row['status']='WITHIN_1M_PREVIOUS_OR_DECLARED_HUB';continue
        row['status']='PENDING_SPATIAL_COVERAGE_SELECTION';eligible.append(row)
    selected=[]
    while eligible and len(selected)<4:
        references=excluded+[point(r['position']) for r in selected]
        # Farthest-point coverage uses only fixed official source coordinates,
        # not visibility labels or runtime success. Lexicographic stable ties.
        eligible.sort(key=lambda r:(-min((math.dist(r['position'],x) for x in references),default=0),r['position']))
        chosen=eligible.pop(0)
        if any(math.dist(chosen['position'],r['position'])<1 for r in selected):
            chosen['status']='WITHIN_1M_NEW_SELECTED_HUB';continue
        chosen['status']='PROSPECTIVE_OFFICIAL_POSITION_LIVE_NAVMESH_AND_WITNESS_UNTESTED';selected.append(chosen)
    for row in eligible:row['status']='NOT_SELECTED_FOUR_HUB_BUDGET'
    return selected,ledger
def main():
    out=HERE/'snapshot_v1';out.mkdir(exist_ok=False);start=time.monotonic()
    fit=set(read(SPLIT)['FIT']);rows=read(MANIFEST);houses=collections.OrderedDict()
    for row in rows:
        if row['house_id'] not in houses:
            assert row['house_id'] in fit and row['split']=='FIT' and row['house_candidate_rank']==0
            houses[row['house_id']]=copy.deepcopy(row)
    assert len(houses)==43
    lock={str(p):sha(p) for p in (MANIFEST,SPLIT,R2R,BASE)};excluded=collections.defaultdict(list);observed=set();observed_hubs=set();exclusion_sources=[]
    for run in CLOSED:
        result=read(run/'result.json');assert result['status']=='SCOUT_CLOSED' and result['error'] is None
        for path in (run/'result.json',run/'STORE_CLOSE_AUDIT.json'):
            lock[str(path)]=sha(path)
        for path in sorted((run/'houses').glob('*/FROZEN_HUB_CONFIGS.json')):
            document=read(path);house=path.parent.name;assert house in houses
            for hub in document['hubs']:
                excluded[house].append(hub['position']);observed.add(house);observed_hubs.add((house,point(hub['position'])))
            lock[str(path)]=sha(path);exclusion_sources.append(str(path))
    # Every prepared certification batch exposes only its immutable hub reset
    # position. Never open its journal/HEAD/PROGRESS or use outcome-based ranks.
    for path in sorted((WF/'batch_execution_v1').glob('batch_*/run_v1/EXECUTION_CONFIG.json')):
        for row in read(path)['candidates']:
            assert row['house_id'] in houses
            excluded[row['house_id']].append(row['configuration']['u_position'])
        lock[str(path)]=sha(path);exclusion_sources.append(str(path))
    for house in excluded:excluded[house]=[list(x) for x in sorted({point(x) for x in excluded[house]})]
    with gzip.open(R2R,'rt') as f:episodes=json.load(f)['episodes']
    provenance={house:collections.defaultdict(list) for house in houses}
    for episode in episodes:
        house=Path(episode['scene_id']).parent.name
        if house not in provenance:continue
        provenance[house][point(episode['start_position'])].append({'episode_id':episode['episode_id'],
            'source_field':'start_position','raw_start_rotation':episode['start_rotation'],'rotation_not_used_for_scout_reset':True})
        for i,pos in enumerate(episode['reference_path']):
            provenance[house][point(pos)].append({'episode_id':episode['episode_id'],'source_field':'reference_path','path_index':i})
    # Manifest order, but first cover houses that have never had a closed scout.
    order=sorted(houses,key=lambda house:(house in observed,list(houses).index(house)))
    jobs=[];inventory=[];ledgers={};maxsource=0
    base=read(BASE)
    for house in order:
        positions=[list(x) for x in sorted(provenance[house])];assert positions
        selected,ledger=select(positions,excluded[house]);ledgers[house]=ledger
        inventory.append({'house_id':house,'scene_group_id':'mp3d:'+house,'split':'FIT',
            'closed_scout_house_before_freeze':house in observed,'official_source_positions':len(positions),
            'excluded_distinct_positions':len(excluded[house]),'selected_source_positions':len(selected),
            'status':'PROSPECTIVE_FOUR_HUB_QUEUE' if len(selected)==4 else 'INSUFFICIENT_DISTINCT_NEW_OFFICIAL_POSITIONS'})
        if len(selected)!=4:continue
        row=copy.deepcopy(houses[house]);assets={}
        for suffix in ('.glb','.house','.navmesh','_semantic.ply'):
            path=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            assets[str(path)]=sha(path)
            if suffix=='.house':assert assets[str(path)]==row['source_house_sha256']
        chosen=[r['position'] for r in selected]
        row.update(source_positions=chosen,assets=assets,scene_glb=next(k for k in assets if k.endswith('.glb')),
            scene_group_id='mp3d:'+house,metadata_permission_roles_are_not_actual_witness_roles=True)
        cfg=copy.deepcopy(base)
        for key in ('main_agent_approval_sha256','source_lock_sha256'):cfg.pop(key,None)
        job_id='scout_'+str(len(jobs)).zfill(3);folder=out/job_id;folder.mkdir()
        cfg.update(node='Q35N_NEW_HUB_SCALE_V1_'+house,candidates=[row],runtime_allowed=False,executable=False,
            training_allowed=False,runtime_adapter_ready=False,main_agent_runtime_approval_required=True,
            gpu_device=7,gpu_uuid=None,shard_id=job_id,intended_output_root=str((folder/'run_v1').relative_to(ROOT)),
            new_hub_plan={'house_id':house,'limit':4,'selected_positions':chosen,'excluded_positions':excluded[house],
                'prior_hub_sources':exclusion_sources,'house_split':'FIT','scene_group_id':'mp3d:'+house,
                'sampling':'official_train_spatial_coverage_not_generalization_evaluation',
                'prior_live_reachability_required_by_cpu_selector':False,'live_geometry_recheck_required':True,
                'original_event_and_component_rules_unchanged':True})
        assert cfg['budget']==dict(total_actions=40000,total_seconds=2700,discovery_actions=40000,
            discovery_seconds=2400,certification_actions=1,certification_seconds=1)
        save(folder/'PREPARED_CONFIG.json',cfg)
        poses=[{'position':pos,'yaw_bin':0,'agent_rotation_wxyz':[1,0,0,0],
            'sensor_positions':{name:[pos[0],pos[1]+1.25,pos[2]] for name in ('rgb','semantic')},
            'status':'REQUESTED_RESET_POSE_NOT_OBSERVED','official_references':provenance[house][point(pos)]} for pos in chosen]
        save(folder/'POSE_PROVENANCE.json',poses)
        jobs.append({'id':job_id,'house_id':house,'proposed_gpu':7,'prepared_config':str(folder/'PREPARED_CONFIG.json'),
            'source_lock':str(folder/'SOURCE_LOCK.json'),'prospective_new_hubs':4,'runtime_allowed':False,
            'runtime_transport_not_implemented_by_this_node':True})
    assert len(jobs)>=30,'FEWER_THAN_THIRTY_HOUSES_HAVE_FOUR_DISTINCT_NEW_SOURCE_POSITIONS'
    queue={'schema_version':'q35n.new_hub_scale_source_queue.v1','jobs':jobs,'houses':len(jobs),
        'prospective_new_hubs':len(jobs)*4,'previous_closed_scout_houses':len(observed),
        'previous_closed_scout_hubs':len(observed_hubs),'selection_uses_runtime_outcomes':False,
        'per_house_worker_seconds':2700,'per_house_supervisor_seconds':3000,'per_house_actions':40000,
        'per_house_content_bytes':6*1024**3,'per_house_disk_bytes':7*1024**3,
        'runtime_transport_and_gpu7_holder_authorization_required':True,'runtime_allowed':False,'scientific_pass':False}
    save(out/'SCOUT_QUEUE.json',queue);save(out/'HOUSE_INVENTORY.json',inventory);save(out/'SELECTION_LEDGER.json',ledgers)
    for path in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',out/'SCOUT_QUEUE.json',out/'HOUSE_INVENTORY.json',out/'SELECTION_LEDGER.json',
        WF/'new_hub_scout_v1/selector.py',WF/'new_hub_scout_v1/adapters.py',WF/'new_hub_scout_v1/runtime.py',WF/'new_hub_scout_v1/SHA256SUMS']:
        lock[str(path)]=sha(path)
    save(out/'SOURCE_LOCK.json',lock)
    for job in jobs:
        folder=Path(job['prepared_config']).parent;cfg=read(folder/'PREPARED_CONFIG.json');merged=dict(lock)
        merged.update(cfg['candidates'][0]['assets'])
        for path in (folder/'PREPARED_CONFIG.json',folder/'POSE_PROVENANCE.json',out/'SOURCE_LOCK.json'):merged[str(path)]=sha(path)
        save(folder/'SOURCE_LOCK.json',merged);maxsource=max(maxsource,len(merged))
    result={'status':'CROSS_HOUSE_NEW_HUB_SOURCES_PREPARED_REQUIRES_NEW_RUNTIME_ADAPTER','houses':len(jobs),
        'previous_unscouted_houses_in_queue':sum(not r['closed_scout_house_before_freeze'] for r in inventory if r['status']=='PROSPECTIVE_FOUR_HUB_QUEUE'),
        'prospective_new_source_positions':4*len(jobs),'new_physically_observed_hubs':0,'new_certified_families':0,
        'max_per_house_source_lock_files':maxsource,'preparation_seconds':time.monotonic()-start,
        'gpu_operations':0,'scientific_pass':False}
    save(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
