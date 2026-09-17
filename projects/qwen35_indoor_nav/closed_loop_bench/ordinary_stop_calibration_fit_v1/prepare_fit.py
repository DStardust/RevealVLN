"""Select and freeze 64 FIT routes before collecting any policy logits."""
import ast
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
REVIEW=LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1'
SNAP=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
PREFIX='Q35N_STOP_CALIBRATION_FIT_V1:'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_SELECTION_ONLY'
    assert not (HERE/'SOURCE_LOCK.json').exists() and not (HERE/'run_001').exists()
    tests=subprocess.run([str(PY),'-I','-S','-B',str(REVIEW/'test_counterfactual.py')],capture_output=True,text=True,timeout=60)
    assert tests.returncode==0,tests.stderr
    r=load('stop_fit_source',HERE/'reuse.py');checks=[]
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
        text=r.source(name);ast.parse(text)
        if name!='aggregate.py':assert text==r.parent.source(name)
        checks.append(name)
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,output=tests.stdout+tests.stderr,
         reused_sources_compiled=checks,policy_simulator_transport_unchanged=True))
    import habitat_sim as hs
    prep=load('stop_fit_route_identity',LINE/'data_pipeline/ordinary_scale_v1/prepare.py')
    official=load('stop_fit_official_distance',BASE/'official_distance.py')
    split=read(SNAP/'SPLIT.json');fit=set(split['FIT'])
    assert fit.isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    src=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
    assert sha(src)=='f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
    groups=collections.defaultdict(list)
    with gzip.open(src) as f:
        for e in json.load(f)['episodes']:
            if e['scene_id'].split('/')[-2] in fit:groups[prep.route_key(e)].append(e)
    pools=collections.defaultdict(list)
    for key,aliases in groups.items():
        e=min(aliases,key=lambda x:int(x['episode_id']));pools[e['scene_id'].split('/')[-2]].append((key,e))
    houses=sorted((h for h,v in pools.items() if len(v)>=4),key=lambda h:hashlib.sha256((PREFIX+h).encode()).hexdigest())[:16]
    assert len(houses)==16
    selected=[]
    for h in houses:
        selected.extend(sorted(pools[h],key=lambda x:hashlib.sha256((PREFIX+x[0]).encode()).hexdigest())[:4])
    selected.sort(key=lambda x:(x[1]['scene_id'],x[0]));houses=sorted(houses)
    assert len(selected)==64 and len({k for k,e in selected})==64
    gtpath=ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_gt.json.gz'
    assert sha(gtpath)=='b63fa10e5c57d1b80b241c49bdeddc3d481bbe54b41ca0d59f9584c10da7bd5d'
    with gzip.open(gtpath) as f:gt=json.load(f)
    pathfinders={};geometry=[];assets=[]
    for index,(key,e) in enumerate(selected):
        house=e['scene_id'].split('/')[-2];assert house in fit and gt[str(e['episode_id'])]['locations']
        if house not in pathfinders:
            pf=hs.PathFinder();nav=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.navmesh'
            assert nav.resolve(strict=True).is_relative_to(ROOT) and pf.load_nav_mesh(str(nav));pathfinders[house]=pf
            assets.extend(nav.with_suffix('.'+ext) for ext in ('navmesh','glb'))
        distance=official.measure(pathfinders[house],e['start_position'],[g['position'] for g in e['goals']],SimpleNamespace(_shortest_path_cache=None))
        assert math.isfinite(distance) and distance>0
        geometry.append(dict(index=index,episode_id=e['episode_id'],house=house,official_start_distance=distance,cached_dataset_distance=e['info']['geodesic_distance']))
    p=read(BASE/'PROTOCOL.json')
    assert p['checkpoint_sha256']=='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    p.update(id='Q35N_ORDINARY_STOP_CALIBRATION_FIT_V1',checkpoint_role='base4000_fit_calibration_only',
             episode_count=64,house_count=16,houses=houses,house_counts={h:4 for h in houses},wall_seconds=3600,
             main_criterion='Calibration data collection only; no navigation improvement claim',
             checkpoint_selection='fixed best pure model 4000; no intermediate selection',controller_enabled=False,
             split='FIT',threshold_fitting_allowed=True)
    save(HERE/'PROTOCOL.json',p)
    save(HERE/'EPISODES_PRIVILEGED.json',[e for k,e in selected])
    save(HERE/'GEOMETRY_PREFLIGHT.json',dict(all_count=64,all_reachable=True,rows=geometry))
    save(HERE/'PARITY_FIXTURES.json',read(BASE/'PARITY_FIXTURES.json'))
    save(HERE/'SELECTION.json',dict(split='FIT',houses=houses,house_counts={h:4 for h in houses},physical_keys=[k for k,e in selected],
         source_sha256=sha(src),gt_sha256=sha(gtpath),input_sha256=sha(HERE/'EPISODES_PRIVILEGED.json'),
         dev_or_confirm_used=False,model_results_used_for_selection=False,not_independent_validation=True))
    files=dict(read(BASE/'SOURCE_LOCK.json')['files'])
    dependencies=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(REVIEW.glob('*.py'))+[REVIEW/'SPEC_ZH.md',SNAP/'SPLIT.json',src,gtpath,LINE/'data_pipeline/ordinary_scale_v1/prepare.py']+assets
    for path in dependencies:
        assert path.resolve(strict=True).is_relative_to(ROOT),path
        files[str(path)]=sha(path)
    for path,digest in files.items():assert sha(path)==digest,path
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='FIT threshold fitting data, no DEV or CONFIRM',training_updates=0))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_BOUNDED_FIT_COLLECTION',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
         gpu=1,requires_empty_gpu=True,borrowed_holders_allowed=False,automatic_retry=False,unix=time.time()))
    print(json.dumps(dict(status='FROZEN',episodes=64,houses=houses,source_files=len(files)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
