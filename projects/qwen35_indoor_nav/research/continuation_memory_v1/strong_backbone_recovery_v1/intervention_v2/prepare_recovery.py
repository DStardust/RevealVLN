"""Freeze new TRAIN route families and the already exposed unseen denominator."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as u


def rank(value):
    return hashlib.sha256(('intervention-recovery-v2:'+str(value)).encode()).hexdigest()


def lock(run):
    previous=u.read(u.HERE/'intervention_runs/calibrate_001/SOURCE_LOCK.json')['files']
    files={p:h for p,h in previous.items() if Path(p).is_relative_to(u.CODE)}
    paths=list(HERE.glob('*.py'))+[u.HERE/n for n in ('common.py','memory_v2.py','transfer_runtime.py','transfer_audit.py',
        'action_boundary_v2.py','intervention_gate_v1.py','transfer_pipeline.py','unseen_pipeline.py','standalone.py')]
    paths += list((run/'fixtures').glob('*'))+[run/n for n in ('PROTOCOL.json','DATA_MANIFEST.json')]
    paths += [run.parent/n for n in ('PROTOCOL.json','UNSEEN_MANIFEST.json','SPLIT_AUDIT.json')]
    p=u.read(run/'PROTOCOL.json')
    for item in list(p['heads'].values())+list(p.get('gates',{}).values()):
        assert u.sha(item['path'])==item['sha256'],'WEIGHT_FILE_CHANGED'
        paths.append(Path(item['path']))
    for path in paths:files[str(path)]=u.sha(path)
    u.write(run/'SOURCE_LOCK.json',dict(files=files))


def main(run_id):
    root=HERE/'runs'/run_id
    assert not root.exists(),'RUN_ALREADY_EXISTS'
    old=u.HERE/'intervention_runs/calibrate_001'
    base=u.read(old/'PROTOCOL.json')
    source=Path(base['source']);source_sha=u.sha(source);raw=json.loads(gzip.decompress(source.read_bytes()))
    excluded={x['house'] for x in u.read(old/'DATA_MANIFEST.json')['episodes']}
    excluded.update(x['house'] for x in u.read(u.HERE/'preservation_runs/preserve_001/DATA_MANIFEST.json')['episodes'])
    memory_config=u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json'
    excluded.update(x['house'] for x in u.read(memory_config)['families'])
    test_manifest=u.HERE/'runs/unseen_001/DATA_MANIFEST.json'
    test_rows=u.read(test_manifest)['episodes']
    unseen_source=u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz'
    unseen=json.loads(gzip.decompress(unseen_source.read_bytes()))['episodes']
    unseen_houses={Path(x['scene_id']).stem for x in unseen}
    pool=defaultdict(lambda:defaultdict(list))
    for ep in raw['episodes']:
        house=Path(ep['scene_id']).stem
        if house in excluded:continue
        if (u.ASSETS/'third_party/ETP-R1/data/scene_datasets'/ep['scene_id']).is_file():
            pool[house][str(ep['trajectory_id'])].append(ep)
    houses=sorted([h for h,r in pool.items() if len(r)>=20],key=rank)[:20]
    assert len(houses)==20 and not set(houses)&unseen_houses
    chosen={h:[min(pool[h][r],key=lambda e:rank(e['episode_id'])) for r in sorted(pool[h],key=lambda r:rank(h+':'+r))[:20]] for h in houses}
    run=root/'train';fixtures=run/'fixtures';fixtures.mkdir(parents=True)
    template=(u.CODE/'config/vln_r2r.yaml').read_text()
    vocab_path=next(k for k in u.read(u.ORIGIN/'FIXTURE_MANIFEST.json')[0]['files'] if k.endswith('.json.gz'))
    vocab=json.loads(gzip.decompress((u.ASSETS/vocab_path).read_bytes()))['instruction_vocab']
    rows=[]
    for route_index in range(20):
        for house_index,h in enumerate(houses):
            ep=chosen[h][route_index]
            route=house_index+20*route_index;data=fixtures/f'{route:03d}.json.gz';config=fixtures/f'{route:03d}.yaml'
            data.write_bytes(gzip.compress(json.dumps(dict(episodes=[ep],instruction_vocab=vocab)).encode(),mtime=0))
            text=template.replace('scenes_dir: data/scene_datasets/',f'scenes_dir: {u.ASSETS}/third_party/ETP-R1/data/scene_datasets/')
            text=text.replace('data_path: data/datasets/r2r/{split}/{split}.json.gz',f'data_path: {data}')
            config.write_text(text.replace('split: val_seen','split: train').replace('habitat:\n','habitat:\n  seed: 42\n',1))
            for variant in ('natural','turn_prefix'):
                prefix=[] if variant=='natural' else [2 if (house_index+route_index)%2==0 else 3]*8
                rows.append(dict(id=len(rows),episode_id=ep['episode_id'],trajectory_id=ep['trajectory_id'],house=h,
                    partition='FIT' if house_index<16 else 'DEV',config=str(config),instruction=ep['instruction']['instruction_text'],
                    variant=variant,forced_prefix=prefix,route_family=f'{h}:{ep["trajectory_id"]}',original_source_sha256=source_sha))
    protocol={k:base[k] for k in ('source_commit','expected_base_state_sha256','backbone','seed','heads')}
    protocol.update(id='TRAIN_INTERVENTION_RECOVERY_V2',planned_groups=len(rows),split='OFFICIAL_TRAIN',gates={},
        source=str(source),source_sha256=u.sha(source),action_rule='Existing frozen real-action residual; both arms execute the registered prefix before autonomous action',
        prefix_rule='Raw/processed input and native argmax agreement through first actual intervention; finite numerical drift recorded',
        perturbation='One natural and one actual 8-turn prefix per route, direction balanced without reading outcomes. All turns count in 500; no teleport, no fabricated observations.',
        forced_action_ids=dict(STOP=0,FORWARD=1,LEFT=2,RIGHT=3),
        old_fit_source=str(old),fit_houses=houses[:16],dev_houses=houses[16:],
        gate=dict(steps=2000,seed=42,optimizer='AdamW lr=.001 weight_decay=.1',architecture='Linear 3592->3',
            input='causal actor feature + centered native logits + current proposal delta',loss='Class-balanced FIT trajectory-outcome CE; no old EU6 action anchors',
            admission='At least one actually observed GAIN and HARM; never relabel a neutral outcome',
            decision='p(GAIN)>2*p(HARM), once at first proposal disagreement, choose entire remaining frozen policy',
            selection='No DEV or unseen parameter/threshold search'),
        checkpoint_limitation='Frozen memory heads previously used EU6, an official val_unseen house. Evaluate only the fixed 200 routes excluding EU6; not a clean full1839 paper model.',
        architecture_prototype='evidence_arch_v1 is separate and is not trained or evaluated by this run')
    u.write(run/'PROTOCOL.json',protocol);u.write(run/'DATA_MANIFEST.json',dict(episodes=rows,source=str(source),source_sha256=u.sha(source)))
    u.write(root/'UNSEEN_MANIFEST.json',dict(episodes=test_rows,source=str(test_manifest),source_sha256=u.sha(test_manifest)))
    fit=set(houses[:16])|set(base['fit_houses']);dev=set(houses[16:])|set(base['dev_houses'])
    assert not fit&dev and not (fit|dev)&unseen_houses
    u.write(root/'SPLIT_AUDIT.json',dict(new_train_routes=400,new_train_physical_conditions=800,new_episodes=3200,
        old_conditions_reused=200,unique_total_routes=600,fit_houses=sorted(fit),dev_houses=sorted(dev),unseen_overlap=[],
        augmented_route_variants_stay_together=True,unseen_planned=200,unseen_houses=sorted({e['house'] for e in test_rows}),
        unseen_previously_exposed=True,full_1839=False,old_head_training_unseen_overlap=['EU6Fwq7SyZv'],
        no_unseen_labels_for_gate=True,outcome_based_route_selection=False,source=str(source),source_sha256=u.sha(source)))
    u.write(root/'PROTOCOL.json',dict(id='INTERVENTION_RECOVERY_V2',training=protocol,unseen_manifest_sha256=u.sha(root/'UNSEEN_MANIFEST.json'),
        gpu_indices=list(range(8)),gpu_session_hours_limit=64,wall_hours_limit=12,artifact_limit_gib=32,
        chunk_pairs=25,first_wave_pairs_per_gpu=2,automatic_score_retries=0,
        phases=['collect_train','fit_gates','live_unseen','review'],
        no_harm_action='DATA_LIMITED when every arm lacks a real class; no fake gate or invented negative label'))
    lock(run)
    u.write(root/'STATUS.json',dict(status='PREPARED',phase='COLLECT_TRAIN',gpu_hours=0,planned_groups=800,recorded_groups=0,sealed_groups=0))
    u.write(u.HERE/'INTERVENTION_V2_RUN.json',dict(path=str(root)))
    print(root)


def prepare_unseen(root):
    train=u.read(root/'train/PROTOCOL.json');review=u.read(root/'GATE_REVIEW.json')
    qualified={a:r for a,r in review['results'].items() if r['status']=='TRAINED'}
    if not qualified:return None
    run=root/'unseen'
    if run.exists():u.verify_sources(run);return run
    run.mkdir();p=dict(train);p.update(id='LIVE_GATED_EXPOSED_UNSEEN_200_V2',split='OFFICIAL_VAL_UNSEEN_SUBSET',planned_groups=200)
    p['heads']=dict(train['heads']);p['gates']={}
    for arm,r in qualified.items():
        p['heads'][arm+'_GATE']=dict(train['heads'][arm],source_arm=arm)
        p['gates'][arm+'_GATE']=dict(path=r['path'],sha256=r['sha256'])
    p['action_rule']='Unmodified ordinary start. One live causal gate choice at first disagreement; real autonomous forwards after branching.'
    p['perturbation']='NONE';p['trained_gate_arms']=list(qualified)
    u.write(run/'PROTOCOL.json',p);manifest=u.read(root/'UNSEEN_MANIFEST.json')
    assert all(not row.get('forced_prefix') for row in manifest['episodes'])
    u.write(run/'DATA_MANIFEST.json',manifest);lock(run)
    # Referenced old fixtures stay read-only and are explicitly locked.
    locked=u.read(run/'SOURCE_LOCK.json')
    for row in manifest['episodes']:
        path=Path(row['config']);locked['files'][str(path)]=u.sha(path)
        for line in path.read_text().splitlines():
            if line.strip().startswith('data_path:'):
                data=Path(line.split(':',1)[1].strip());locked['files'][str(data)]=u.sha(data)
    u.write(run/'SOURCE_LOCK.json',locked)
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);main(parser.parse_args().run_id)
