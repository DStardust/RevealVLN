"""Register real FIT-only assets and scale before collection or any model scores."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    inventory=read(V16/'DATA_MANIFEST.json');old=read(CPU/'DATA.json')
    natural=read(PARENT.parent/'natural_transfer_v9/DATA.json')
    holdout=read(PARENT/'monotonic_holdout_v1/HOUSE_MANIFEST.json')
    excluded=set(natural['check_houses'])
    excluded|={h['house'] for h in inventory['houses'] if h['split']!='FIT'}
    excluded|={h['house'] for h in holdout['houses']}
    excluded|={f['house'] for f in old['raw_families'] if f['split']!='FIT'}
    # Exactly the thirteen unconsumed entries plus the four original FIT houses.
    houses=inventory['available_unexposed_houses'][13:]+[h['house'] for h in inventory['houses'] if h['split']=='FIT']
    if len(houses)!=17 or set(houses)&excluded:raise ValueError('HOUSE_ISOLATION')
    old_positions={}
    for f in old['raw_families']:
        old_positions.setdefault(f['house'],[])
        if f['initial_position'] not in old_positions[f['house']]:old_positions[f['house']].append(f['initial_position'])
    planner=load('scale_metadata',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    scene_root=Path(inventory['houses'][0]['scene']).parent.parent;records=[]
    for h in houses:
        assets=[scene_root/h/(h+s) for s in ('.glb','.house','.navmesh','_semantic.ply')]
        if not all(p.is_file() for p in assets):raise FileNotFoundError(str(assets))
        groups=planner.semantic_groups(planner.parse_house(assets[1].read_text()))
        roles=[dict(signature=list(k),spec=planner.role_spec(k),eligible=[o['object_index'] for o in v]) for k,v in sorted(groups.items())]
        records.append(dict(house=h,split='FIT',scene=str(assets[0]),roles=roles,asset_sha256={str(p):sha(p) for p in assets}))
    immutable(HERE/'HOUSE_MANIFEST.json',dict(houses=records,excluded=sorted(excluded),old_positions=old_positions,
        selection='13 previously unused inventory entries and four original FIT houses, selected without model scores.',
        base_exposure='Not claimed unseen by best4k/foundation training. New houses are now FIT; cannot be used as hidden TEST.',
        sources={str(p.relative_to(LINE)):sha(p) for p in (V16/'DATA_MANIFEST.json',CPU/'DATA.json',PARENT.parent/'natural_transfer_v9/DATA.json',PARENT/'monotonic_holdout_v1/HOUSE_MANIFEST.json')}))
    oldcfg=read(PARENT/'monotonic_holdout_v1/PROTOCOL.json')
    cfg={k:oldcfg[k] for k in ('asset_line_root','sim_python','standalone_python','publication_branch','seed','devices')}
    cfg.update(version='EVENT_DATA_SCALE20_V1',houses=houses,families_per_house=64,target_new_parents=800,
        planned_parent_capacity=1088,planned_variant_capacity=2176,baseline_FIT_parents=32,baseline_total_parents=40,
        positions_per_house=1536,plans_per_position=12,delay_actions=[8,16,32,64],turn_counts=[2,4,6,8,10,12],turn_directions=['L','R'],
        parent_min_distance_m=1.,max_session_hours=24,max_wall_hours=48,gpu_session_hours=384,infra_retries=2,
        min_start_gpu_gib=4,min_free_gpu_gib=2,process_rss_gib=24,artifact_gib=300,per_house_artifact_gib=17,
        scope='Stationary SEE2 event recognition/history retention/STOP/reorientation; no translational recovery or ordinary VLN gain claim.',
        no_training=True,no_Qwen_forward=True,base_updates=0,
        completion='Finish the fixed 17-house collection. >=800 audited new parents meets scale request; fewer reports physical/resource shortfall. Counterparts/windows/labels never increase parent count.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    immutable(HERE/'PROTOCOL.json',cfg)
    paths={LINE/p for p in read(PARENT/'monotonic_holdout_v1/SOURCE_LOCK.json')['files']}
    paths.update(HERE.glob('*.py'));paths.update([HERE/'PROTOCOL.json',HERE/'HOUSE_MANIFEST.json',HERE/'index.html'])
    paths.add(PILOT/'parallel_eval_v1/coordinator.py')
    immutable(HERE/'SOURCE_LOCK.json',dict(source_commit=cfg['source_commit'],files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)}))
    print(dict(houses=houses,target=800,capacity=1088,training=False),flush=True)

if __name__=='__main__':main()

