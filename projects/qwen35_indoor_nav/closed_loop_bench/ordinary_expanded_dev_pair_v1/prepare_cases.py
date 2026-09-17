"""CPU-only deterministic development selection and per-case checkpoint binding."""
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'r2r_ce_full_v2';TRAIN=LINE/'sft_acceptance/ordinary_expanded_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def select():
    import habitat_sim as hs
    prep=load('source_route_identity',LINE/'data_pipeline/ordinary_scale_v1/prepare.py')
    official=load('original_official_distance',OLD/'official_distance.py')
    split=read(LINE/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json')
    path=ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz'
    assert sha(path)=='f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34'
    episodes=json.load(gzip.open(path))['episodes'];grouped=collections.defaultdict(list)
    for e in episodes:
        if e['scene_id'].split('/')[-2] in split['INTERNAL_DEV']:grouped[prep.route_key(e)].append(e)
    pools=collections.defaultdict(list)
    for key,aliases in grouped.items():
        e=min(aliases,key=lambda x:int(x['episode_id']));pools[e['scene_id'].split('/')[-2]].append((key,e))
    for house in pools:pools[house].sort(key=lambda item:hashlib.sha256(('Q35N_EXPANDED_DEV_V1:'+item[0]).encode()).hexdigest())
    selected=[]
    for rank in range(max(map(len,pools.values()))):
        for house in sorted(pools):
            if rank<len(pools[house]):selected.append(pools[house][rank])
            if len(selected)==100:break
        if len(selected)==100:break
    assert len(selected)==100 and len({x[0] for x in selected})==100
    selected.sort(key=lambda item:(item[1]['scene_id'],item[0]))
    es=[x[1] for x in selected];houses=sorted(pools)
    assert set(houses)==set(split['INTERNAL_DEV']) and set(houses).isdisjoint(split['FIT']+split['INTERNAL_CONFIRM'])
    gtpath=ROOT/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_gt.json.gz'
    gt=json.load(gzip.open(gtpath));paths={};geometry=[]
    for index,e in enumerate(es):
        house=e['scene_id'].split('/')[-2]
        assert str(e['episode_id']) in gt and gt[str(e['episode_id'])]['locations']
        if house not in paths:
            pf=hs.PathFinder();nav=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.navmesh'
            assert nav.resolve(strict=True).is_relative_to(ROOT) and pf.load_nav_mesh(str(nav));paths[house]=pf
        d=official.measure(paths[house],e['start_position'],[g['position'] for g in e['goals']],SimpleNamespace(_shortest_path_cache=None))
        assert math.isfinite(d) and d>0
        geometry.append(dict(index=index,episode_id=e['episode_id'],house=house,official_start_distance=d,
                             cached_dataset_distance=e['info']['geodesic_distance']))
    save(HERE/'EPISODES_PRIVILEGED.json',es)
    save(HERE/'GEOMETRY_PREFLIGHT.json',dict(all_count=100,all_reachable=True,rows=geometry))
    save(HERE/'SELECTION.json',dict(source=str(path),source_sha256=sha(path),gt_path=str(gtpath),gt_sha256=sha(gtpath),
         houses=houses,house_counts=dict(collections.Counter(e['scene_id'].split('/')[-2] for e in es)),
         physical_keys=[x[0] for x in selected],pool_capacities={h:len(v) for h,v in pools.items()},
         input_sha256=sha(HERE/'EPISODES_PRIVILEGED.json'),selection_depends_on_model_results=False,
         split='INTERNAL_DEV',official_val_unseen_used=False))
    save(HERE/'SELECTION_SEAL.json',{p.name:sha(p) for p in (HERE/'EPISODES_PRIVILEGED.json',HERE/'GEOMETRY_PREFLIGHT.json',HERE/'SELECTION.json',HERE/'SPEC_ZH.md')})
    print(json.dumps(read(HERE/'SELECTION.json'),ensure_ascii=False))


def bind_case(role,checkpoint):
    # Called after CPU selection; GPU process not yet started.
    for name,digest in read(HERE/'SELECTION_SEAL.json').items():assert sha(HERE/name)==digest
    case=HERE.parent/f'ordinary_expanded_dev_{role}_v1';selection=read(HERE/'SELECTION.json')
    assert role in ('before','after') and checkpoint.resolve(strict=True).is_relative_to(LINE)
    receipt=read(str(checkpoint)+'.json');assert sha(checkpoint)==receipt['sha256']
    if role=='before':
        assert receipt['cursor']['updates']==51301 and receipt['sha256']=='4d60cdb09a909121856e438547df5c6e1e9ea35fc40a8f257302bf563af4979e'
        tp=LINE/'sft_acceptance/ordinary_sync_recovery_v1/PROTOCOL_FILESTORE.json'
    else:
        tp=TRAIN/'PROTOCOL_FILESTORE.json'
        final=read(TRAIN/'formal/attempt_001/RESULT.json')
        assert checkpoint==Path(final['latest_checkpoint']) and final['cursor']['updates']>=200
        lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
        assert lease['execute_returned'] and lease['holders_restored'] and final['status'] in ('STOPPED','EPOCHS_COMPLETED')
        assert final['stop']==['BUDGET:max_updates'], 'PILOT_DID_NOT_REACH_PLANNED_UPDATES'
    p=read(OLD/'PROTOCOL.json')
    p.update(id='Q35N_EXPANDED_DEV_'+role.upper()+'_V1',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_protocol_sha256=sha(tp),sample_index_sha256=read(tp)['sample_index_sha256'],seed=read(tp)['seed'],
        checkpoint_updates=receipt['cursor']['updates'],checkpoint_role=role,episode_count=100,house_count=5,
        houses=selection['houses'],house_counts=selection['house_counts'],gt_path=selection['gt_path'],
        wall_seconds=3600,output_gib=4,previous_tiny_episode_ids=[],
        main_criterion='Paired development diagnostic; not scientific PASS',checkpoint_selection='fixed before 51301 / final bounded pilot after')
    for k in ('checkpoint_selected_unix',):p.pop(k,None)
    save(case/'PROTOCOL.json',p)
    for name,source in [('EPISODES_PRIVILEGED.json',HERE),('GEOMETRY_PREFLIGHT.json',HERE),('PARITY_FIXTURES.json',OLD)]:save(case/name,read(source/name))
    tests=read(HERE/'CPU_TEST_RESULT.json');assert tests['passed'];save(case/'CPU_TEST_RESULT.json',tests)
    files=dict(read(OLD/'SOURCE_LOCK.json')['files'])
    for path in list(case.glob('*.py'))+list(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md',HERE/'SELECTION_SEAL.json',tp,checkpoint,Path(str(checkpoint)+'.json'),
             case/'PROTOCOL.json',case/'EPISODES_PRIVILEGED.json',case/'GEOMETRY_PREFLIGHT.json',case/'PARITY_FIXTURES.json',Path(selection['gt_path'])]:
        files[str(path)]=sha(path)
    for house in p['houses']:
        for ext in ('glb','navmesh'):
            path=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{house}/{house}.{ext}'
            assert path.resolve(strict=True).is_relative_to(ROOT);files[str(path)]=sha(path)
    save(case/'SOURCE_LOCK.json',dict(files=files,scope='fixed same-input paired INTERNAL_DEV evaluation',training_updates=0))
    return case


if __name__=='__main__':select()
