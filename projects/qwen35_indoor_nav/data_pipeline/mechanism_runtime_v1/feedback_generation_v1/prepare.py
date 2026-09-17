import gzip
import importlib.util
import json
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent
LINE=RUNTIME.parents[1]
ROOT=LINE.parents[1]
def sha(p):
    p=p.resolve(strict=True);assert p.is_relative_to(ROOT)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    manifest=LINE/'data_pipeline/mechanism_scale_v1/acceptance_v1/candidates.json'
    rows=json.loads(manifest.read_text())[:3]
    assert len({r['house_id'] for r in rows})==3 and all(r['house_candidate_rank']==0 for r in rows)
    split=LINE/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
    groups=json.loads(split.read_text())
    source=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
    assert sha(source)=='f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
    with gzip.open(source,'rt') as f:episodes=json.load(f)['episodes']
    for row in rows:
        house=row['house_id'];assert house in groups['FIT']
        row['assets']={}
        for suffix in ('.glb','.house','.navmesh','_semantic.ply'):
            p=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}{suffix}'
            row['assets'][str(p)]=sha(p)
        row['scene_glb']=next(p for p in row['assets'] if p.endswith('.glb'))
        positions={tuple(p) for e in episodes if Path(e['scene_id']).parent.name==house for p in [e['start_position']]+e['reference_path']}
        row['source_positions']=[list(p) for p in sorted(positions)]
        for task,anchor in [('task_A','anchor_A'),('task_B','anchor_B')]:
            def desc(role):
                r=row['roles'][role];room={'tv':'TV room','familyroom/lounge':'family room/lounge'}.get(r['room'],r['room'])
                return 'the '+r['raw_match']['value']+' in the '+room
            row['tasks'][task]=dict(anchor=anchor,terminal='terminal',
                instruction='First see '+desc(anchor)+' in two consecutive observations, then see '+desc('terminal')+' in two consecutive observations, and stop immediately.')
    cfg=dict(node='Q35N_FEEDBACK_MECHANISM_GENERATION_V1',runtime_allowed=True,gpu_device=1,
        gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',candidates=rows,
        source_sha256=sha(source),manifest_sha256=sha(manifest),split_sha256=sha(split),
        budget=dict(total_actions=40000,total_seconds=2700,discovery_actions=12000,
                    discovery_seconds=600,certification_actions=20000,certification_seconds=600),
        supervision_wall_seconds=3000,training_allowed=False,scientific_pass=False)
    with (out/'EXECUTION_CONFIG.json').open('x') as f:json.dump(cfg,f,indent=2)
    paths=[HERE/x for x in ('feedback.py','store.py','worker.py','run.py','prepare.py','SPEC_ZH.md')]
    paths += [RUNTIME/x for x in ('core_bridge.py','habitat_backend.py','guard.py','runtime_journal.py','exporter.py','loader.py')]
    paths += list((LINE/'data_pipeline/mechanism_factory_v2').glob('*.py'))
    paths += [LINE/'MAINLINE_FREEZE_V3.md',manifest,split,source,out/'EXECUTION_CONFIG.json']
    with (out/'INPUT_LOCK.json').open('x') as f:json.dump({str(p):sha(p) for p in paths},f,indent=2)
    print(json.dumps(dict(prepared=True,candidates=3,houses=[r['house_id'] for r in rows],runtime_allowed=True)))
if __name__=='__main__':main()

