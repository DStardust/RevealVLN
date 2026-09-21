"""Freeze new houses from asset metadata and existing exposure, before any scores."""
import collections
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    old=read(PARENT/'gpu_runtime_r1/PROTOCOL.json')
    inventory=read(V16/'DATA_MANIFEST.json')
    # First four unused entries after the nine previously registered V16 houses.
    houses=inventory['available_unexposed_houses'][9:13]
    if len(houses)!=4:raise ValueError('FOUR_NEW_HOUSES_REQUIRED')
    natural=read(PARENT.parent/'natural_transfer_v9/DATA.json')
    excluded=set(natural['fit_houses'])|set(natural['check_houses'])|set(inventory['exposure'])
    excluded|={h['house'] for h in inventory['houses']}
    sources={}
    for path in sorted(V16.rglob('FAMILY.json')):
        family=read(path)
        if 'house' in family:excluded.add(family['house']);sources[str(path.relative_to(LINE))]=sha(path)
    data=read(CPU/'DATA.json')
    excluded|={f['house'] for f in data['raw_families']}
    if set(houses)&excluded:raise ValueError('NEW_HOUSE_EXPOSURE:'+str(set(houses)&excluded))
    planner=load('holdout_metadata',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    scene_root=Path(inventory['houses'][0]['scene']).parent.parent
    records=[]
    for house in houses:
        folder=scene_root/house
        assets=[folder/(house+suffix) for suffix in ('.glb','.house','.navmesh','_semantic.ply')]
        if not all(p.is_file() for p in assets):raise FileNotFoundError(str(assets))
        groups=planner.semantic_groups(planner.parse_house(assets[1].read_text()))
        roles=[dict(signature=list(k),spec=planner.role_spec(k),eligible=[r['object_index'] for r in v]) for k,v in sorted(groups.items())]
        records.append(dict(house=house,scene=str(assets[0]),roles=roles,split='TEST',asset_sha256={str(p):sha(p) for p in assets}))
    immutable(HERE/'HOUSE_MANIFEST.json',dict(houses=records,excluded=sorted(excluded),exposure_sources=sources,
        selection='First four entries [9:13] of frozen V16 available asset order; disjoint from registered nine, current memory data, ordinary pool and actual prior V16 FAMILY records.',
        base_training_exposure='Unresolved: new memory-evaluation houses, not claimed unseen by best4k or its foundation model.',
        language='Same DEV instruction templates, new house/physical families; not independent natural-language validation.'))
    devices=[dict(gpu=int(line.split(',')[0]),gpu_uuid=line.split(',')[1].strip()) for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()]
    cfg={k:old[k] for k in ('asset_line_root','torch_python','sim_python','standalone_python','checkpoint','checkpoint_sha256',
        'model_source_sha256','training_protocol_sha256','sample_index_sha256','seed','seeds','model_memory_gib','process_rss_gib',
        'warmup_indices','warmup_repeats','publication_branch')}
    cfg.update(version='MONOTONIC_HOLDOUT_V1',houses=houses,families_per_house=4,arms=['DIRECT','MONOTONIC'],devices=devices,
        gpu_session_hours=20,max_session_hours=3,min_start_gpu_gib=12,min_free_gpu_gib=2,artifact_gib=40,
        positions_per_house=96,plans_per_position=8,delay_actions=[8,16,32],turn_counts=[2,4,6,8,10,12],turn_directions=['L','R'],
        planned_variants=32,planned_conditions=256,planned_rollouts=1536,main_denominator_per_arm=384,control_denominator_per_arm=384,
        source_models=str(PARENT/'gpu_v1/runs/gpu_001'),new_training_updates=0,base_updates=0,infra_retries=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary='MONOTONIC minus DIRECT safe SEE2 task_A success; four houses equally weighted, three frozen seeds equally weighted',
        endpoints='Report task_T, seen/missing history, terminal present/absent, collisions and full-denominator budget cost separately.',
        decision='Positive lower identification bound with positive direction in >=3/4 houses and >=2/3 seeds supports a limited new-house controlled-task signal. No automatic deployment or CVPR claim.',
        missing='All 32 planned variant slots retained. Physically exhausted proposals are NOT_COLLECTED, never collision FAIL or dropped N.',
        selection='No new checkpoint selection, training, threshold search or method-score access during collection. First certified parent per position; first four parents per house.',
        scope='Stationary SEE2 stopping/reorientation. Same task semantics and templates; no natural VLN-CE or translational recovery claim.',
        authorization='User: execute next step; all GPUs available, standalone continuation and restore own placeholders.')
    immutable(HERE/'PROTOCOL.json',cfg)
    prior=read(PARENT/'gpu_runtime_r1/SOURCE_LOCK.json')['files']
    paths=set(LINE/p for p in prior)
    paths.update(HERE.glob('*.py'))
    paths.update([HERE/'HOUSE_MANIFEST.json',HERE/'PROTOCOL.json',LINE/'data_pipeline/mechanism_scale_v1/planner.py'])
    for sub in ('identifiable_data_v1','b2_state_coverage_v1'):
        paths.update((V16/sub).glob('*.py'))
    paths.update((V16/'identifiable_data_v1/balanced_probe_001').glob('*.json'))
    immutable(HERE/'SOURCE_LOCK.json',dict(source_commit=cfg['source_commit'],files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)}))
    print(dict(houses=houses,planned=cfg['planned_rollouts'],new_updates=0))

if __name__=='__main__':main()
