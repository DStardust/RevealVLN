"""Second score-blind FIT batch, registered separately from the preserved 599-parent batch."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main():
    prior=PARENT/'event_data_scale20_v1'
    cfg=read(prior/'PROTOCOL.json');inventory=read(V16/'DATA_MANIFEST.json')
    natural=read(PARENT.parent/'natural_transfer_v9/DATA.json');old=read(prior/'HOUSE_MANIFEST.json')
    # Conservative exclusion: all previous mechanism-data/semantic-pool houses, even if old files are now unavailable.
    prior_memory={h for h,paths in inventory['exposure'].items() if any(
        'multifamily_v7' in p or 'semantic_transfer_v12' in p or ('/DATA.json' in p and 'natural_transfer_v9' not in p) for p in paths)}
    excluded=set(old['excluded'])|set(cfg['houses'])|prior_memory|set(natural['check_houses'])
    eligible=[h for h in natural['fit_houses'] if h not in excluded]
    houses=eligible[:16]
    if len(houses)!=16:raise ValueError('NOT_ENOUGH_KNOWN_FIT_ASSETS')
    planner=load('scale2_metadata',LINE/'data_pipeline/mechanism_scale_v1/planner.py')
    scene_root=Path(inventory['houses'][0]['scene']).parent.parent;records=[]
    for h in houses:
        assets=[scene_root/h/(h+s) for s in ('.glb','.house','.navmesh','_semantic.ply')]
        if not all(p.is_file() for p in assets):raise FileNotFoundError(str(assets))
        groups=planner.semantic_groups(planner.parse_house(assets[1].read_text()))
        roles=[dict(signature=list(k),spec=planner.role_spec(k),eligible=[o['object_index'] for o in v]) for k,v in sorted(groups.items())]
        records.append(dict(house=h,split='FIT',scene=str(assets[0]),roles=roles,asset_sha256={str(p):sha(p) for p in assets}))
    baseline_files={}
    for p in sorted((prior/'runs/scale_001/collect').glob('*/position_*/FAMILY.json')):
        baseline_files[str(p.relative_to(LINE))]=sha(p)
    if len(baseline_files)!=599:raise ValueError('PRIOR_PARENT_COUNT_CHANGED')
    immutable(HERE/'HOUSE_MANIFEST.json',dict(houses=records,old_positions={},excluded=sorted(excluded),
        selection='First 16 eligible houses in frozen ordinary FIT order; no candidate model scores read.',
        base_exposure='Known ordinary FIT houses, not claimed unseen by the base policy. Only new physical SEE2 families count.',
        ordinary_fit_houses=natural['fit_houses'],ordinary_check_houses=natural['check_houses'],
        prior_memory_excluded=sorted(prior_memory),
        sources={str(p.relative_to(LINE)):sha(p) for p in (prior/'HOUSE_MANIFEST.json',V16/'DATA_MANIFEST.json',PARENT.parent/'natural_transfer_v9/DATA.json')},
        prior_parent_files=baseline_files))
    cfg.update(version='EVENT_DATA_SCALE20_V2_SUPPLEMENT',houses=houses,families_per_house=32,
        prior_run=str(prior/'runs/scale_001'),prior_unit='q35n-scale20-data-r2-20260922-01.service',
        prior_pipeline=str(prior/'pipeline_r2.py'),prior_pipeline_sha256=sha(prior/'pipeline_r2.py'),
        target_combined_parents=800,registered_prior_parents=599,minimum_additional_parents=201,
        planned_parent_capacity=512,planned_variant_capacity=1024,per_house_artifact_gib=17,
        completion='At >=800 combined audited parents, launch no further houses; finish all already-started houses and prior variants. Preserve unstarted registered houses as NOT_NEEDED_AFTER_TARGET. No threshold relaxation.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    immutable(HERE/'PROTOCOL.json',cfg)
    paths={LINE/p for p in read(prior/'SOURCE_LOCK.json')['files']}
    paths.update(LINE/p for p in read(prior/'RUNTIME_AMENDMENT_R2.json')['files'])
    paths.update(HERE.glob('*.py'));paths.update([HERE/'PROTOCOL.json',HERE/'HOUSE_MANIFEST.json',HERE/'index.html'])
    immutable(HERE/'SOURCE_LOCK.json',dict(files={str(p.relative_to(LINE)):sha(p) for p in sorted(paths)},source_commit=cfg['source_commit']))
    print(dict(houses=houses,additional_capacity=512,combined_target=800),flush=True)

if __name__=='__main__':main()

