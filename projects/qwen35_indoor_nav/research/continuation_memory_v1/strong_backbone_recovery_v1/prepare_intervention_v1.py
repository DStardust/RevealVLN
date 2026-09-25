"""Outcome-independent, route-unique new TRAIN calibration collection."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def rank(value):return hashlib.sha256(('20260925:intervention:'+str(value)).encode()).hexdigest()


def main(run_id):
    root=u.HERE/'intervention_runs'/run_id;assert not root.exists(),'REGISTERED_RUN_EXISTS'
    parent=u.HERE/'preservation_runs/preserve_001';old=u.read(parent/'unseen/PROTOCOL.json')
    source=u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz'
    source_sha=u.sha(source);raw=json.loads(gzip.decompress(source.read_bytes()))
    excluded={r['house'] for r in u.read(parent/'DATA_MANIFEST.json')['episodes']}
    excluded.update(f['house'] for f in u.read(u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json')['families'])
    unseen_houses={r['house'] for r in u.read(parent/'UNSEEN_MANIFEST.json')['episodes']}
    pool=defaultdict(lambda:defaultdict(list))
    for ep in raw['episodes']:
        house=ep['scene_id'].split('/')[-2]
        if house in excluded:continue
        scene=u.ASSETS/'third_party/ETP-R1/data/scene_datasets'/ep['scene_id']
        if scene.is_file():pool[house][str(ep['trajectory_id'])].append(ep)
    houses=sorted([h for h,r in pool.items() if len(r)>=20],key=rank)[:10];assert len(houses)==10
    assert not set(houses)&unseen_houses
    chosen={h:[min(pool[h][route],key=lambda e:rank(str(e['episode_id'])+':instruction')) for route in sorted(pool[h],key=lambda r:rank(h+':'+r))[:20]] for h in houses}
    root.mkdir(parents=True);fixtures=root/'fixtures';fixtures.mkdir()
    template=(u.CODE/'config/vln_r2r.yaml').read_text();vocab_path=next(k for k in u.read(u.ORIGIN/'FIXTURE_MANIFEST.json')[0]['files'] if k.endswith('.json.gz'))
    vocab=json.loads(gzip.decompress((u.ASSETS/vocab_path).read_bytes()))['instruction_vocab'];rows=[]
    for i in range(20):
        for h in houses:
            ep=chosen[h][i];idx=len(rows);data=fixtures/f'{idx:03d}.json.gz';config=fixtures/f'{idx:03d}.yaml'
            data.write_bytes(gzip.compress(json.dumps(dict(episodes=[ep],instruction_vocab=vocab)).encode(),mtime=0))
            text=template.replace('scenes_dir: data/scene_datasets/',f'scenes_dir: {u.ASSETS}/third_party/ETP-R1/data/scene_datasets/')
            text=text.replace('data_path: data/datasets/r2r/{split}/{split}.json.gz',f'data_path: {data}')
            text=text.replace('split: val_seen','split: train').replace('habitat:\n','habitat:\n  seed: 42\n',1);config.write_text(text)
            rows.append(dict(id=idx,episode_id=ep['episode_id'],trajectory_id=ep['trajectory_id'],house=h,
                partition='FIT' if h in houses[:8] else 'DEV',config=str(config),instruction=ep['instruction']['instruction_text'],original_source_sha256=source_sha))
    protocol={k:old[k] for k in ['source_commit','expected_base_state_sha256','backbone','seed','action_rule','prefix_rule','order','heads']}
    protocol.update(id='TRAIN_INTERVENTION_CALIBRATION_V1',purpose='Learn whether to keep native or switch to the frozen method at its first actual disagreement',
        dataset_source='OFFICIAL_TRAIN',source=str(source),source_sha256=source_sha,
        planned_groups=200,planned_episodes=800,fit_routes=160,dev_routes=40,fit_houses=houses[:8],dev_houses=houses[8:],
        gpu_indices=list(range(8)),gpu_session_hours_limit=16,wall_hours_limit=6,
        base_updates=0,memory_head_updates=0,gate_updates_per_arm=2000,gate_seed=42,
        gate=dict(type='Linear three-class selector on frozen causal actor features and current proposal logits',
            optimizer='AdamW lr .001 weight_decay .1, full FIT batch, fixed 2000 updates',
            selection='p(SR gain)>2*p(SR harm), one decision at first disagreement for the entire remaining policy',
            loss='Natural-frequency trajectory outcome CE + .5 known-FIT-action retention margin',
            task_anchors='Separate 10 admitted debug FIT action points, NOT labelled as navigation successes',
            no_score_search=True,no_unseen_training=True),
        task_fit_pack=str(u.HERE/'repair_runs/action_001/data/FIT.pt'),
        dataset_semantics='A paired rollout labels complete policy continuation value, not each changed action. Same outcomes are neutral, not proof of state equivalence.',
        final_review='DEV selects between already executed complete branches once. This is not a newly executed online gated policy, nor unseen SR.',
        data_gap='If either gain or harm is absent in FIT for an arm, report missing supervision; do not invent labels or silently add test cases.',
        old_unseen='Prior 200-route outcome analysis is diagnosis only. No old unseen route/label is included.',
        generalization='These are new adapter calibration houses from official TRAIN; public backbone may already have seen all TRAIN houses.')
    u.write(root/'PROTOCOL.json',protocol);u.write(root/'DATA_MANIFEST.json',dict(episodes=rows,source=str(source),source_sha256=source_sha))
    u.write(root/'SPLIT_AUDIT.json',dict(fit_houses=houses[:8],dev_houses=houses[8:],excluded_old_ordinary_and_memory_houses=sorted(excluded),
        source_train=True,unique_routes=len({(r['house'],r['trajectory_id']) for r in rows}),unseen_house_overlap=[],outcome_based_selection=False,
        ranking='SHA256 20260925:intervention namespace; first 10 available houses with >=20 unique routes, 20 per house, first8FIT/last2DEV'))
    files={path:digest for path,digest in u.read(parent/'unseen/SOURCE_LOCK.json')['files'].items() if Path(path).is_relative_to(u.CODE)}
    for path in [u.HERE/n for n in ('common.py','transfer_runtime.py','transfer_worker_v2.py','transfer_audit.py','action_boundary_v2.py',
            'memory_v2.py','unseen_pipeline.py','transfer_pipeline.py','standalone.py','intervention_pipeline_v1.py','intervention_gate_v1.py','prepare_intervention_v1.py','test_intervention_v1.py')]+list(fixtures.iterdir())+[root/'PROTOCOL.json',root/'DATA_MANIFEST.json',root/'SPLIT_AUDIT.json',Path(protocol['task_fit_pack'])]:
        files[str(path)]=u.sha(path)
    for head in protocol['heads'].values():assert u.sha(head['path'])==head['sha256'];files[head['path']]=head['sha256']
    u.write(root/'SOURCE_LOCK.json',dict(files=files))
    u.write(root/'STATUS.json',dict(status='PREPARED',phase='COLLECT_TRAIN_PAIRS',planned_groups=200,planned_episodes=800,recorded_groups=0,sealed_groups=0,gpu_hours=0))
    u.write(u.HERE/'INTERVENTION_RUN.json',dict(path=str(root)))
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(a.run_id)
