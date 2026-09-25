"""Freeze a score-independent official unseen subset, excluding memory FIT houses."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def pick(episodes,count,seed,excluded_houses,excluded_routes):
    pools=defaultdict(lambda:defaultdict(list))
    for episode in episodes:
        house=episode['scene_id'].split('/')[-2];route=str(episode['trajectory_id'])
        if house not in excluded_houses and (house,route) not in excluded_routes:pools[house][route].append(episode)
    total=sum(len(routes) for routes in pools.values())
    if not 0<count<=total:raise ValueError('INSUFFICIENT_DISTINCT_ROUTES')
    quotas={h:count*len(routes)//total for h,routes in pools.items()}
    remaining=count-sum(quotas.values())
    for h in sorted(pools,key=lambda h:(-(count*len(pools[h])%total),h))[:remaining]:quotas[h]+=1
    def rank(house,value):return hashlib.sha256(f'{seed}|{house}|{value}'.encode()).hexdigest()
    chosen={}
    for house,routes in sorted(pools.items()):
        ordered=sorted(routes,key=lambda r:rank(house,r))[:quotas[house]]
        chosen[house]=[min(routes[r],key=lambda ep:rank(house,str(ep['episode_id'])+'|instruction')) for r in ordered]
    rows=[]
    for i in range(max(quotas.values())):
        for h in sorted(chosen):
            if i<len(chosen[h]):rows.append(chosen[h][i])
    return rows,dict(eligible_distinct_routes=total,house_route_counts={h:len(p) for h,p in pools.items()},house_quotas=quotas)


def main(run,count,seed):
    run.mkdir(parents=True,exist_ok=False)
    source=u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz'
    raw=json.loads(gzip.decompress(source.read_bytes()));assert len(raw['episodes'])==1839
    dev_source=u.PROJECT/'sft_acceptance/ordinary_branch_decision_v21/runs/pilot_002/DATA_MANIFEST.json'
    dev=[x['episode'] for x in u.read(dev_source)['evaluation']]
    dev_routes={(ep['scene_id'].split('/')[-2],str(ep['trajectory_id'])) for ep in dev}
    # Whole memory-training house excluded, not just the ten fitted actions.
    memory_source=u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json'
    memory_fit={f['house'] for f in u.read(memory_source)['families'] if f['partition']=='fit'}
    rows,audit=pick(raw['episodes'],count,seed,memory_fit,dev_routes)
    vocab_fixture=u.read(u.ORIGIN/'FIXTURE_MANIFEST.json')[0]
    vocab_path=next(k for k in vocab_fixture['files'] if k.endswith('.json.gz'))
    vocab=json.loads(gzip.decompress((u.ASSETS/vocab_path).read_bytes()))['instruction_vocab']
    template=(u.CODE/'config/vln_r2r.yaml').read_text();fixtures=run/'fixtures';fixtures.mkdir();manifest=[]
    for index,episode in enumerate(rows):
        house=episode['scene_id'].split('/')[-2];data_path=fixtures/f'{index:03d}.json.gz';config_path=fixtures/f'{index:03d}.yaml'
        assert (u.ASSETS/'third_party/ETP-R1/data/scene_datasets'/episode['scene_id']).is_file()
        data_path.write_bytes(gzip.compress(json.dumps(dict(episodes=[episode],instruction_vocab=vocab)).encode(),mtime=0))
        config=template.replace('scenes_dir: data/scene_datasets/',f'scenes_dir: {u.ASSETS}/third_party/ETP-R1/data/scene_datasets/')
        config=config.replace('data_path: data/datasets/r2r/{split}/{split}.json.gz',f'data_path: {data_path}')
        config=config.replace('split: val_seen','split: val_unseen').replace('habitat:\n','habitat:\n  seed: 42\n',1)
        config_path.write_text(config)
        manifest.append(dict(id=index,episode_id=episode['episode_id'],trajectory_id=episode['trajectory_id'],house=house,
            config=str(config_path),instruction=episode['instruction']['instruction_text'],original_source_sha256=u.sha(source)))
    baseline=u.HERE/'runs/native_001';protocol=u.read(baseline/'PROTOCOL.json')
    protocol.update(id='Q35N_STREAMVLN_UNSEEN_SUBSET_V1',evaluation='Official val_unseen fixed 200-route subset, excluding memory FIT house and old development routes',
        status='REGISTERED_NATIVE_BASELINE',native_episodes=count,identity_pairs=0,identity_evidence=str(baseline/'RESULT.json'),
        sampling_seed=seed,gpu_indices=[0,1],wall_hours_limit=3,gpu_session_hours_limit=6,
        selection='House-proportional largest-remainder quotas, SHA256 route order, one fixed instruction per distinct trajectory; no outcome input',
        only_native_arm=True,method_training_completed=True,method_evaluation_in_this_run=False,
        training_on_this_subset_forbidden=True,previous_public_split_exposed=True,blind_generalization_claim=False)
    u.write(run/'PROTOCOL.json',protocol)
    u.write(run/'DATA_MANIFEST.json',dict(episodes=manifest,source=str(source),source_sha256=u.sha(source),original_results_reused=False))
    selected_routes={(x['house'],str(x['trajectory_id'])) for x in manifest}
    assert len(selected_routes)==count and not selected_routes&dev_routes
    u.write(run/'SPLIT_AUDIT.json',dict(source_episodes=1839,selected_episodes=count,selected_unique_routes=len(selected_routes),
        excluded_memory_fit_houses=sorted(memory_fit),excluded_memory_house_episodes=sum(e['scene_id'].split('/')[-2] in memory_fit for e in raw['episodes']),
        development_overlap_routes=0,development_houses=sorted({h for h,_ in dev_routes}),selected_houses=sorted({h for h,_ in selected_routes}),
        development_source_sha256=u.sha(dev_source),memory_source_sha256=u.sha(memory_source),**audit,
        outcome_based_selection=False,scope='Public exposed benchmark subset; not a replacement for missing history-task CHECK.'))
    u.write(run/'ASSET_PROVENANCE.json',u.read(baseline/'ASSET_PROVENANCE.json'))
    files={path:digest for path,digest in u.read(baseline/'SOURCE_LOCK.json')['files'].items() if Path(path).is_relative_to(u.CODE)}
    for p in [u.HERE/n for n in ('unseen_pipeline.py','native_worker.py','common.py','memory.py','prepare_unseen.py')]+list(fixtures.iterdir())+[run/'PROTOCOL.json',run/'DATA_MANIFEST.json',run/'SPLIT_AUDIT.json']:
        files[str(p)]=u.sha(p)
    u.write(run/'SOURCE_LOCK.json',dict(files=files,upstream_commit=protocol['source_commit']))
    u.write(run/'STATUS.json',dict(status='PREPARED',phase='UNSEEN_BASELINE',planned_native=count,completed_native=0,recorded_native=0))
    u.write(u.HERE/'UNSEEN_RUN.json',dict(run=run.name,path=str(run)))
    print(json.dumps(u.read(run/'SPLIT_AUDIT.json'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--count',type=int,default=200);p.add_argument('--seed',type=int,default=1209)
    a=p.parse_args();main(u.HERE/'runs'/a.run_id,a.count,a.seed)
