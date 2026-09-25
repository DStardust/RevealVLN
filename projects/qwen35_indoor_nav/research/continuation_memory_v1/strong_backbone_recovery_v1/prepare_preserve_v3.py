"""Register train-only ordinary preservation and freeze unseen without scores."""
import argparse
from collections import defaultdict,Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def main(run_id):
    root=u.HERE/'preservation_runs'/run_id
    assert not (root/'PROTOCOL.json').exists(),'REGISTERED_RUN_IS_READ_ONLY'
    old=u.HERE/'repair_runs/action_001';runtime=u.read(old/'ordinary/PROTOCOL.json')
    source=u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz'
    train={str(x['episode_id']):x for x in json.load(gzip.open(source,'rt'))['episodes']}
    manifest=u.read(u.HERE/'runs/native_001/DATA_MANIFEST.json');rows=manifest['episodes'];by_house=defaultdict(list);routes=set();files={}
    for row in rows:
        fixture=Path(row['config']).with_suffix('.json.gz');episode=json.load(gzip.open(fixture,'rt'))['episodes'][0]
        original=train[str(episode['episode_id'])]
        assert episode['instruction']['instruction_text']==original['instruction']['instruction_text']
        for key in ('start_position','start_rotation','trajectory_id','scene_id','goals'):
            assert episode[key]==original[key],('NOT_EXACT_TRAIN_EPISODE',row['id'],key)
        route=(row['house'],episode['trajectory_id']);assert route not in routes,'DUPLICATE_ROUTE_REQUIRES_GROUP_SPLIT'
        routes.add(route);row['trajectory_id']=episode['trajectory_id'];by_house[row['house']].append(row)
        files[str(fixture)]=u.sha(fixture);files[row['config']]=u.sha(row['config'])
    assert len(rows)==100
    for house,group in by_house.items():
        ordered=sorted(group,key=lambda r:hashlib.sha256(f"42:{house}:{r['trajectory_id']}".encode()).hexdigest())
        n=round(len(group)*.2)
        for rank,row in enumerate(ordered):row['partition']='DEV' if rank<n else 'FIT'
    dev=[r for r in rows if r['partition']=='DEV'];assert len(dev)==20
    unseen=u.read(u.HERE/'runs/unseen_001/DATA_MANIFEST.json')
    assert not {r['house'] for r in rows}&{r['house'] for r in unseen['episodes']}
    for row in unseen['episodes']:
        for path in (Path(row['config']),Path(row['config']).with_suffix('.json.gz')):files[str(path)]=u.sha(path)
    inherited={k:runtime[k] for k in ('source_commit','expected_base_state_sha256','backbone','seed','action_rule','prefix_rule','order')}
    protocol=dict(id='STREAMVLN_ORDINARY_PRESERVATION_V3',memory_run=str(old),physical_data=str(u.HERE/'data_runs/moving_003'),
        base_state_sha256=runtime['expected_base_state_sha256'],runtime=inherited,preservation_weights=[5.,20.,80.],
        gpu_indices=list(range(8)),gpu_session_hours_limit=24,wall_hours_limit=12,
        ordinary_fit_trajectories=80,ordinary_dev_trajectories=20,unseen_planned_groups=200,
        ordinary_targets='Actual frozen native actions, all outcomes included, no oracle or success filtering',
        teacher='Actual complete collision-free visible successful FIT suffix; shortest actual execution per history/task',
        loss='Inherited sparse task objective + lambda*(unmasked ordinary KL + native-action margin) + .5 ordinary CE + .5 FIT class-balanced dense suffix CE',
        optimizer=dict(seed=42,steps=1200,learning_rate=1e-4,weight_decay=.01,clip_norm=1,base_updates=0),
        architecture='Unchanged memory_v2.ExecutionMemory 8x64, 3584-dimensional frozen StreamVLN features',
        selection='First registered weight with at least one B2/Ours head retaining 10/10 special FIT actions and DEV paired SR >= native; all three heads evaluated. Cached fidelity descriptive, not a gate.',
        rejection_of_trivial_fix='Zero head is 0/10 and constant STOP 8/10 on retained task points; neither qualifies',
        unseen='Fixed exposed unseen200 evaluated once after DEV selection; no retraining or weight selection from unseen scores',
        task_fit_scope='One admitted FIT family/two contrasts; three FIT families can provide dense legal suffix actions. Old training_admission=false unchanged; not formal task generalization.',
        retry='Infrastructure-only resume from optimizer checkpoint/complete sealed sessions; no score-based episode retries',
        stopping='At most three strengths, then one full unseen comparison or explicit DEV failure; no automatic architecture or task changes',
        proof_needed='Empirical aggregate SR >= same-session native and nontrivial task fit; never a universal no-harm guarantee')
    root.mkdir(parents=True,exist_ok=True)
    u.write(root/'PROTOCOL.json',protocol);u.write(root/'DATA_MANIFEST.json',manifest)
    u.write(root/'DEV_MANIFEST.json',dict(episodes=dev));u.write(root/'UNSEEN_MANIFEST.json',unseen)
    u.write(root/'SPLIT_AUDIT.json',dict(source=str(source),source_sha256=u.sha(source),exact_train_scene_start_goal_text_route_matches=100,
        legacy_instruction_token_ids='Differ by legacy vocabulary versus XLMR; StreamVLN retokenizes the same instruction_text with its frozen tokenizer; unused IDs never enter its input.',
        unique_routes=100,split_rule='House-stratified SHA256(42:house:trajectory_id), first rounded 20% DEV, no score input',
        counts=dict(Counter(r['partition'] for r in rows)),houses={h:dict(Counter(r['partition'] for r in g)) for h,g in by_house.items()},
        unseen_house_overlap=[],memory_fit_house_overlap=[],formerly_exposed_native100=True,blind=False))
    capture=root/'capture';capture.mkdir();u.write(capture/'PROTOCOL.json',dict(inherited,heads={},capture_only=True))
    u.write(capture/'DATA_MANIFEST.json',manifest)
    names=['common.py','memory_v2.py','action_boundary_v2.py','transfer_runtime.py','transfer_worker_v2.py','transfer_audit.py',
        'transfer_pipeline.py','unseen_pipeline.py','train_memory_v2.py','review_memory_r1.py','capture_runtime_v3.py',
        'capture_worker_v3.py','preserve_data_v3.py','preserve_objective_v3.py','preserve_train_v3.py',
        'preserve_review_v3.py','preserve_pipeline_v3.py','prepare_preserve_v3.py','test_preserve_v3.py','standalone.py']
    files.update({str(u.HERE/name):u.sha(u.HERE/name) for name in names})
    for path in (root/'PROTOCOL.json',root/'DATA_MANIFEST.json',root/'DEV_MANIFEST.json',root/'UNSEEN_MANIFEST.json',
                 capture/'PROTOCOL.json',capture/'DATA_MANIFEST.json',old/'data/FIT.pt',old/'features/FEATURE_INDEX.json'):
        files[str(path)]=u.sha(path)
    inherited_lock=u.read(old/'ordinary/SOURCE_LOCK.json')['files']
    # Preserve provenance of the official inference stack without binding unrelated old runners.
    files.update({path:sha for path,sha in inherited_lock.items() if not Path(path).is_relative_to(u.HERE)})
    lock=dict(files=files,old_corrected_runtime_source_lock_sha256=u.sha(old/'SOURCE_LOCK.json'))
    u.write(root/'SOURCE_LOCK.json',lock);u.write(capture/'SOURCE_LOCK.json',lock)
    u.write(root/'STATUS.json',dict(status='PREPARED',phase='CPU_VALIDATION',captured_recorded=0,gpu_hours=0))
    u.write(u.HERE/'PRESERVATION_RUN.json',dict(path=str(root),protocol=str(root/'PROTOCOL.json')))
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(a.run_id)
