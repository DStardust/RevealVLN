"""Freeze first three new FIT houses now; preserve next 34-house inventory."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
LINE=ROOT/'projects/qwen35_indoor_nav'
MANIFEST=LINE/'data_pipeline/mechanism_scale_v1/acceptance_v1/candidates.json'
SPLIT=LINE/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
R2R=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
BASE=WF/'scout_v1/PREPARED_CONFIG.json'
CLOSED=[WF/'scout_v1/run_v1/result.json',WF/'scout_next_v1/shard_0/run_v1/result.json']

def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);assert path.is_relative_to(HERE);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def choose(rows,fit,excluded):
    ordered=[];seen=set()
    for row in rows:
        house=row['house_id']
        if house in seen:continue
        seen.add(house);assert row['split']=='FIT' and house in fit and row['house_candidate_rank']==0
        if house not in excluded:ordered.append(copy.deepcopy(row))
    return ordered
def positions(episodes,house):
    matched=[e for e in episodes if Path(e['scene_id']).parent.name==house]
    values=sorted({tuple(p) for e in matched for p in [e['start_position']]+e['reference_path']})
    assert matched and values
    return [list(p) for p in values],len(matched)

def main():
    assert not (HERE/'SOURCE_LOCK.json').exists(),'NO_OVERWRITE'
    started=time.monotonic();closed_docs=[read(p) for p in CLOSED]
    assert all(d['status']=='SCOUT_CLOSED' and d['error'] is None for d in closed_docs)
    excluded=[r['house_id'] for d in closed_docs for r in d['houses'] if r['status']=='SCOUT_COMPONENT_BANK_COMPLETE']
    assert len(excluded)==len(set(excluded))==6
    rows=read(MANIFEST);fit=set(read(SPLIT)['FIT']);assert len({r['house_id'] for r in rows})==43
    remaining=choose(rows,fit,set(excluded));assert len(remaining)==37
    old_pending=read(WF/'scout_next_v1/shard_1/PREPARED_CONFIG.json')
    old_houses=[r['house_id'] for r in old_pending['candidates']]
    assert {r['house_id'] for r in remaining[:3]}==set(old_houses)
    assert not (WF/'scout_next_v1/shard_1/run_v1').exists(),'OLD_PENDING_SHARD_ALREADY_EXECUTED'
    with gzip.open(R2R,'rt') as f:episodes=json.load(f)['episodes']
    lock={str(p):sha(p) for p in [MANIFEST,SPLIT,R2R,BASE,*CLOSED,WF/'scout_next_v1/shard_1/PREPARED_CONFIG.json',
          LINE/'data_pipeline/DUAL_PRODUCTION_TRAINING_SCALE_V2.md']}
    inventory=[];source_positions={};ready=[]
    for index,row in enumerate(remaining):
        house=row['house_id'];pos,episode_count=positions(episodes,house)
        source_positions[house]={'house_id':house,'scene_group_id':'mp3d:'+house,'split':'FIT',
            'source':'official_R2R_CE_train_v1_3','source_episodes':episode_count,'positions':pos}
        assets={};stat_rows=[]
        for suffix in ('.glb','.house','.navmesh','_semantic.ply'):
            path=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            assert path.is_file() and path.resolve()==path
            h=sha(path) if index<3 or suffix=='.house' else None
            if h is not None:lock[str(path)]=h
            assets[str(path)]=h;stat_rows.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h})
            if suffix=='.house':assert h==row['source_house_sha256']
        row.update(source_positions=pos,assets=assets,scene_glb=next(p for p in assets if p.endswith('.glb')),
            scene_group_id='mp3d:'+house,source_house_order=index,
            metadata_permission_roles_are_not_actual_witness_roles=True)
        status='FIRST_SLICE_CPU_ASSET_LOCKED_NOT_EXECUTABLE' if index<3 else 'REMAINING_INVENTORY_ASSET_HASHING_DEFERRED'
        inventory.append({'house_id':house,'source_order':index,'scene_group_id':'mp3d:'+house,'split':'FIT',
            'source_episodes':episode_count,'source_position_count':len(pos),'assets':stat_rows,'status':status,
            'old_prepared_unexecuted_shard1_overlap':house in old_houses,'physical_hubs_generated':0,'families_generated':0})
        if index<3:ready.append(row)
    save(HERE/'INVENTORY_ALL_37.json',{'rows':inventory,'excluded_completed_houses':excluded,
        'selection_rule':'original_manifest_first_encounter_minus_six_closed_houses_no_runtime_outcome_ranking',
        'original_houses':43,'remaining_houses':37,'first_slice_houses':3,'deferred_houses':34,
        'planned_first_wave_houses':18,'planned_first_wave_shards':6,
        'old_pending_shard1_preserved':True,'old_pending_shard1_must_not_execute_in_parallel_with_overlap':old_houses,
        'training_admission':False,'scientific_pass':False})
    save(HERE/'SOURCE_POSITIONS.json',source_positions)
    spec=importlib.util.spec_from_file_location('bulk_source_adapters',HERE/'adapters.py')
    adapters=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapters)
    base=read(BASE)
    for shard in range(1):
        folder=HERE/f'shard_{shard:02d}';cfg=copy.deepcopy(base)
        cfg.update(node=f'Q35N_BULK_SCOUT_FIRST_WAVE_V1_{shard:02d}',candidates=ready[shard*3:shard*3+3],
            runtime_allowed=False,executable=False,training_allowed=False,gpu_device=1,gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',
            shard_id=shard,intended_output_root=str((folder/'run_v1').relative_to(ROOT)),
            main_agent_runtime_approval_required=True,runtime_adapter_ready=False,
            split_sha256=lock[str(SPLIT)],source_sha256=lock[str(R2R)],manifest_sha256=lock[str(MANIFEST)],
            scene_group_policy='all_same_house_routes_tasks_roles_seeds_branches_stay_in_same_FIT_group')
        cfg.pop('source_configuration_sha256',None)
        assert cfg['budget']['total_actions']==60000 and cfg['budget']['total_seconds']==4200
        save(folder/'PREPARED_CONFIG.json',cfg);save(folder/'TRANSPORT_DRAFT.json',adapters.transport_record(shard))
        for name,fn in [('SCOUT_COMMON_DRAFT.py',adapters.common_source),('SCOUT_WORKER_GPU1_DRAFT.py',adapters.worker_source),
                        ('BANK_RECIPE_DRAFT.py',adapters.recipe_source),('SUPERVISOR_DRAFT.py',adapters.supervisor_source)]:
            source=fn(shard);compile(source,str(folder/name),'exec')
            with (folder/name).open('x') as f:f.write(source)
        for p in folder.iterdir():lock[str(p)]=sha(p)
    previous=read(WF/'scout_v1/INPUT_LOCK.json')
    for relative,h in previous.items():
        if relative.endswith('.py') or relative.endswith('/SHA256SUMS'):
            path=ROOT/relative;assert sha(path)==h;lock[str(path)]=h
    for p in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SCENE_GROUP_SCHEMA.json',HERE/'INVENTORY_ALL_37.json',
              HERE/'SOURCE_POSITIONS.json',WF/'winding_balance_v1/next_source_v1/prepare.py',
              WF/'winding_balance_v1/method.py']:
        lock[str(p)]=sha(p)
    save(HERE/'SOURCE_LOCK.json',lock)
    save(HERE/'result.json',{'status':'CPU_PREPARED_NOT_EXECUTABLE','houses_asset_locked':3,'shards':1,
        'houses_inventory_deferred':34,'source_files_locked':len(lock),'maximum_prepared_hubs':6,
        'all_43_house_two_hub_upper_bound':86,'remaining_37_house_two_hub_upper_bound':74,
        'actual_hubs_generated':0,'actual_families_generated':0,'gpu_operations':0,'training_admission':False,
        'scientific_pass':False,'preparation_wall_seconds':time.monotonic()-started})
    print(json.dumps(read(HERE/'result.json'),indent=2))

if __name__=='__main__':main()
