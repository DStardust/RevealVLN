"""Freeze real train/dev/unseen registries before collection or model scoring."""
import collections,gzip,json,random,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u

def main():
    assert not (u.HERE/'PROTOCOL.json').exists(),'ALREADY_FROZEN'
    source=u.ASSET.parents[1]/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr'
    train=json.load(gzip.open(source/'train/train.json.gz'))['episodes'];unseen=json.load(gzip.open(source/'val_unseen/val_unseen.json.gz'))['episodes']
    dev=u.read(u.ASSET/'closed_loop_bench/ordinary_cycle_pair_gpu1_v3/EPISODES_PRIVILEGED.json')[:100]
    house=lambda e:e['scene_id'].split('/')[-2]
    excluded={house(e) for e in dev}|{house(e) for e in unseen}
    oldfit=u.read(u.ASSET/'closed_loop_bench/ordinary_stop_calibration_fit_v2/EPISODES_PRIVILEGED.json')
    oldroutes={(house(e),str(e['trajectory_id'])) for e in oldfit}
    byhouse=collections.defaultdict(list)
    for i,e in enumerate(train):
        h=house(e)
        if h in excluded or (h,str(e['trajectory_id'])) in oldroutes:continue
        byhouse[h].append((i,e))
    houses=sorted(h for h,rs in byhouse.items() if len({str(e['trajectory_id']) for i,e in rs})>=8);random.Random(1209).shuffle(houses);houses=houses[:40];assert len(houses)==40
    chosen=[];proposals=[]
    for h in sorted(houses):
        rows=byhouse[h].copy();random.Random('coverage15:'+h).shuffle(rows);seen=set();selected=[]
        for i,e in rows:
            key=str(e['trajectory_id'])
            if key in seen:continue
            seen.add(key);selected.append((i,e))
            if len(selected)==8:break
        assert len(selected)==8
        for i,e in selected:chosen.append(e);proposals.append(dict(source_index=i,episode_id=e['episode_id'],trajectory_id=e['trajectory_id'],house=h))
    assert len(chosen)==320 and len(unseen)==1839 and len(dev)==100
    assert {house(e) for e in chosen}.isdisjoint({house(e) for e in dev}|{house(e) for e in unseen})
    files={};missing=[]
    for h in sorted({house(e) for e in chosen+dev+unseen}):
        for ext in ('glb','navmesh'):
            p=u.ASSET.parents[1]/f'third_party/ETP-R1/data/scene_datasets/mp3d/{h}/{h}.{ext}'
            if not p.is_file():missing.append(str(p))
            else:files[str(p)]=dict(size=p.stat().st_size)
    assert not missing,missing
    folder=u.HERE/'manifests';folder.mkdir()
    for tag,rows in [('FIT',chosen),('DEV',dev),('UNSEEN',unseen)]:u.write(folder/(tag+'_EPISODES.json'),rows,True)
    fitorder=[dict(rank=2*i+j,index=i,episode_id=e['episode_id'],house=house(e),mode=mode,inference_order=['C']) for i,e in enumerate(chosen) for j,mode in enumerate(('POLICY','TEACHER'))]
    devorder=u.read(u.V5/'PAIR_ORDER.json')
    # House blocks reduce loading; order is frozen without any model scores.
    ids=sorted(range(len(unseen)),key=lambda i:(house(unseen[i]),str(unseen[i]['episode_id'])))
    unorder=[dict(rank=r,index=i,episode_id=unseen[i]['episode_id'],house=house(unseen[i]),inference_order=['A','B'] if r%2==0 else ['B','A']) for r,i in enumerate(ids)]
    for tag,rows in [('FIT',fitorder),('DEV',devorder),('UNSEEN',unorder)]:u.write(folder/(tag+'_ORDER.json'),rows,True)
    diag=sorted(houses);random.Random(1210).shuffle(diag)
    u.write(folder/'SPLIT.json',dict(fit_houses=diag[8:],diagnostic_houses=diag[:8],all_training_houses=sorted(houses),dev_houses=sorted({house(e) for e in dev}),unseen_houses=sorted({house(e) for e in unseen}),old64_routes_excluded=True),True)
    index=u.ASSET/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/TRAINING_INDEX.jsonl'
    base_houses={r['scene_group'] for r in u.c.records(index)}
    overlap=sorted(base_houses & {house(e) for e in unseen})
    u.write(folder/'BASE_TRAIN_SCENE_AUDIT.json',dict(source=str(index),sha256=u.sha(index),known_base_training_houses=sorted(base_houses),val_unseen_overlap=overlap),True)
    assert not overlap,'BASE_TRAIN_UNSEEN_OVERLAP'
    u.write(folder/'ASSET_AUDIT.json',dict(chosen=proposals,assets=files,missing=missing,source_dataset_sha256={split:u.sha(source/split/(split+'.json.gz')) for split in ('train','val_unseen')},planned_physical_trajectories=640,paired_instructions=320,unseen_exposure='Already exposed public validation; no blind generalization claim'),True)
    p=u.read(u.HERE.parent/'ordinary_stop_boundary_v14/PROTOCOL.json')
    for k in list(p):
        if k.startswith('pair_') or k.startswith('ranking') or k in ('fit_data','change','loss','milestones','pairs','output_gib'):p.pop(k,None)
    devices=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).strip().splitlines()]
    p.update(id='Q35N_ORDINARY_STOP_COVERAGE_V15',semantic_version='coverage15.1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=u.ROOT,text=True).strip(),
        devices=[dict(gpu=int(i),gpu_uuid=uuid) for i,uuid in devices if 1<=int(i)<=7],
        change='New V13 on-policy and actual route-teacher FIT coverage; original V13 anchored BCE recipe, no ranking loss.',
        loss='house/trajectory equal weighted binary STOP margin BCE + .001/2 L2; normalized causal features; zero residual over best4k; no class reweight/search',
        regularization=.001,model_memory_gib=8,cpu_memory_gib=64,wall_seconds=86400,gpu_hours_limit=48,artifact_gib=400,
        collection_episodes=640,training_instructions=320,training_houses=40,collection_max_steps=500,
        eval_phases=['DEV','UNSEEN'],eval_counts=dict(DEV=100,UNSEEN=1839),unseen_runs_once_without_score_selection=True,
        manifests=str(folder),train_gt=str(source/'train/train_gt.json.gz'),unseen_gt=str(source/'val_unseen/val_unseen_gt.json.gz'),
        candidate_name='V15_REAL_COVERAGE',environment_lanes=2,
        order='Frozen manifest: collection and unseen house blocks, DEV inherits V5; balanced AB/BA within every evaluation pair',
        completion='collect -> CPU train -> DEV100 pairs -> full val_unseen1839 pairs -> review, no automatic deployment')
    u.write(u.HERE/'PROTOCOL.json',p,True);(u.HERE/'LAST_RUN.txt').write_text('coverage_001\n')
    print(json.dumps(dict(train_houses=40,instructions=320,collection_trajectories=640,dev_pairs=100,unseen_pairs=1839)))
if __name__=='__main__':main()
