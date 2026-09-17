"""CPU-only all-episode official distance preflight. No renderer/model/GPU context."""
import collections
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import habitat_sim as hs
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');official=load('official_distance')


def main():
    assert not (HERE/'GEOMETRY_PREFLIGHT.json').exists()
    previous=HERE.parent/'r2r_ce_full_v1'
    episodes=json.loads((previous/'EPISODES_PRIVILEGED.json').read_text())
    paths={};rows=[]
    for index,e in enumerate(episodes):
        house=e['scene_id'].split('/')[-2]
        if house not in paths:
            pf=hs.PathFinder();navmesh=c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.navmesh'
            assert pf.load_nav_mesh(str(navmesh));paths[house]=pf
        pf=paths[house]
        episode=SimpleNamespace(_shortest_path_cache=None)
        distance=official.measure(pf,e['start_position'],[g['position'] for g in e['goals']],episode)
        assert math.isfinite(distance) and distance>0
        repeat=official.measure(pf,e['start_position'],[g['position'] for g in e['goals']],episode)
        assert distance==repeat
        original=e['info']['geodesic_distance']
        rows.append(dict(index=index,episode_id=e['episode_id'],house=house,official_start_distance=distance,
            cached_dataset_distance=original,cache_delta=distance-original,
            start_snap_distance=float(np.linalg.norm(pf.snap_point(np.asarray(e['start_position'],dtype=np.float32))-e['start_position'])),
            goal_snap_distance=float(np.linalg.norm(pf.snap_point(np.asarray(e['goals'][0]['position'],dtype=np.float32))-e['goals'][0]['position']))))
    mismatches=[x for x in rows if abs(x['cache_delta'])>=.05]
    # Audit the already reported tiny trajectories with the exact official
    # distance service too. Never overwrite their original reports.
    tiny=[];tiny_episodes=json.loads((c.TINY/'EPISODES_PRIVILEGED.json').read_text())
    for i,e in enumerate(tiny_episodes):
        saved=json.loads((c.TINY/'run_001'/f'episode_{i:02d}.json').read_text())
        pf=paths[saved['house']];ep=SimpleNamespace(_shortest_path_cache=None)
        distances=[official.measure(pf,position,[g['position'] for g in e['goals']],ep) for position in saved['positions']]
        success=float(saved['stopped'] and distances[-1]<3)
        length=sum(math.dist(a,b) for a,b in zip(saved['positions'],saved['positions'][1:]))
        spl=success*distances[0]/max(distances[0],length)
        tiny.append(dict(episode_id=e['episode_id'],success=success,old_success=saved['success'],spl=spl,old_spl=saved['spl'],
            distance_max_difference=max(abs(a-b) for a,b in zip(distances,saved['distances'])),
            old_metric_agrees=success==saved['success'] and math.isclose(spl,saved['spl'],rel_tol=1e-5,abs_tol=1e-5)))
    result=dict(all_count=len(rows),all_reachable=True,all_repeated_queries_identical=True,cache_mismatches_ge_005=len(mismatches),
        mismatch_houses=dict(collections.Counter(x['house'] for x in mismatches)),
        max_cache_delta=max(abs(x['cache_delta']) for x in rows),
        official_distance_source_sha256=c.sha(official.SOURCE),rows=rows,tiny_reaudit=tiny,
        conclusion='Official runtime distance differs from dataset cached info; cached info is not used by official SR/SPL. Runtime must match this frozen official CPU preflight, not that cache.')
    c.write(HERE/'GEOMETRY_PREFLIGHT.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
