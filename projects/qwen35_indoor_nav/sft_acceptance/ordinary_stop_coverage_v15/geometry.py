"""CPU-only navmesh preflight for the registered official distance backend."""
import json,math,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import habitat_sim as hs
def main(run):
    metric=u.load('coverage15_distance',u.ASSET/'closed_loop_bench/ordinary_cycle_pair_gpu1_v3/official_distance.py');cfg=u.read(run/'PROTOCOL.json');folder=Path(cfg['manifests']);cache={}
    fingerprints={}
    for path,meta in u.read(folder/'ASSET_AUDIT.json')['assets'].items():
        asset=Path(path);assert asset.stat().st_size==meta['size'],'SCENE_SIZE_CHANGED'
        fingerprints[path]=dict(size=meta['size'],sha256=u.sha(asset))
    u.write(run/'SCENE_IDENTITY.json',fingerprints)
    for tag in ('FIT','DEV','UNSEEN'):
        target=run/(tag+'_GEOMETRY.json')
        if target.exists():continue
        rows=[]
        for index,e in enumerate(u.read(folder/(tag+'_EPISODES.json'))):
            house=e['scene_id'].split('/')[-2]
            if house not in cache:
                path=u.ASSET.parents[1]/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.navmesh';pf=hs.PathFinder();assert pf.load_nav_mesh(str(path));cache[house]=pf
            d=metric.measure(cache[house],e['start_position'],[e['goals'][0]['position']],SimpleNamespace(_shortest_path_cache=None));assert math.isfinite(d) and d>0,'INVALID_GEOMETRY'
            rows.append(dict(index=index,episode_id=e['episode_id'],distance=d,source_cached_distance=e['info']['geodesic_distance']))
        u.write(target,rows,True)
    u.write(run/'GEOMETRY_RESULT.json',dict(status='COMPLETE',houses=len(cache),new_gpu_calls=0),True)
if __name__=='__main__':main(Path(sys.argv[1]))
