"""Freeze one training change and its same-process old/new unseen comparison."""
import argparse
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ADAPT = HERE.parent
BASE = ADAPT.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u


def training(root):
    from objective import legacy
    legacy.torch.set_num_threads(2)
    old = ADAPT / 'runs/train_001/TRAINING_CONFIG.json'
    config = u.read(old)
    rows = legacy.load_rows(config['capture_run'], 'FIT')
    if sorted(r['id'] for r in rows) != config['fit_ids']:
        raise ValueError('FIT_DATA_CHANGED')
    audit = {}
    for kind in ('RECOVERY', 'PRESERVATION'):
        selected = [r for r in rows if r['kind'] == kind]
        classes = {}
        for scope in ('ALL', 'FIRST'):
            counts = Counter()
            for r in selected:
                mask = r['known'] & ((r['action_token_offsets'] == 0) if scope == 'FIRST' else legacy.torch.ones_like(r['known']))
                counts.update(r['targets'][mask].tolist())
            classes[scope] = dict(counts)
        if any(not bool((r['known'] & (r['action_token_offsets'] == 0)).any()) for r in selected):
            raise ValueError('EMPTY_FIRST_SUPERVISION')
        audit[kind] = dict(trajectories=len(selected),classes=classes)
    source = dict(config['source_files'])
    source.update({str(HERE/n):u.sha(HERE/n) for n in ('objective.py','train_worker.py')})
    config.update(version='FIRST_ALIGNED_TRAINING_V1',action_scope='FIRST_ACTION_TOKENS_QUERY_START_MEMORY',
        source_files=source,gpu_memory_limit_gib=2.,old_training_config_path=str(old),old_training_config_sha256=u.sha(old),
        only_training_change='Actor loss selects offset zero; complete physical memory unroll remains unchanged.',
        class_weights_unchanged=True,initialization='Same saved untrained initialization, not old FINAL.',
        test_used_for_training=False)
    config['input_files'][str(old)] = u.sha(old)
    root.mkdir(parents=True,exist_ok=False)
    train = root/'training';train.mkdir()
    u.write(train/'TRAINING_CONFIG.json',config)
    u.write(root/'DATA_AUDIT.json',dict(by_kind=audit,fit_trajectories=len(rows),
        memory_updates_unchanged=True,raw_training_admission_unchanged=True,unseen_in_loss=False))
    discovery = ADAPT/'scope_validation_v1/runs/scope_001/unseen'
    manifest = u.read(discovery/'DATA_MANIFEST.json')
    u.write(root/'EVALUATION_MANIFEST.json',manifest)
    operation = u.read(ADAPT/'first_confirmation_v1/runs/confirm_001/PROTOCOL.json')
    operation.update(id='FIRST_ALIGNED_TRAINING_AND_UNSEEN_V1',training_run=str(train),
        planned_groups=80,planned_executions=1040,phases=['TRAIN_SHARED','WAIT_EVALUATION_DEVICE','FIRST_UNSEEN_GROUP','UNSEEN','REVIEW'],
        phase_gpu_hour_limits=dict(TRAIN=4.,EVALUATE=24.),wall_hours_limit=12,
        inherited_phase_gpu_hours=dict(TRAIN=0.,EVALUATE=0.),inherited_wall_seconds=0.,
        first_complete_group_id=manifest['episodes'][0]['id'],chunk_groups=4,
        training_minimum_free_mib=6144,training_gpu_memory_limit_gib=2.,
        allowed_shared_service='q35n-strong-first-confirmation-20260926-01.service',
        discovery_protocol=str(discovery/'PROTOCOL.json'),
        effect='Aligned versus old all-position training, same FIRST inference and matched seed/mode.',
        final_selection=False,optimizer_updates=18000,
        common_changes='FIRST-only supervision in both CURRENT and DELTA; no architecture or runtime change.',
        source_training_config_sha256=u.sha(train/'TRAINING_CONFIG.json'),
        training_admission='FIT_ONLY_FIRST_POSITION_VIEW_OF_SEALED_REAL_DATA',
        revision='Optimization branch requested by user; do not wait for old efficacy scores to train.')
    u.write(root/'PROTOCOL.json',operation)
    files = dict(config['source_files'],**config['input_files'])
    for p in HERE.glob('*.py'):files[str(p)] = u.sha(p)
    for p in [root/'PROTOCOL.json',root/'EVALUATION_MANIFEST.json',train/'TRAINING_CONFIG.json',discovery/'PROTOCOL.json']:
        files[str(p)] = u.sha(p)
    u.write(root/'SOURCE_LOCK.json',dict(files=files))
    print(str(root))


def evaluation(root):
    import torch
    op = u.read(root/'PROTOCOL.json');train=Path(op['training_run']);config=u.read(train/'TRAINING_CONFIG.json')
    p = u.read(op['discovery_protocol'])
    old_heads={f'OLD_{a}':dict(r,training_action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY')
        for a,r in p['heads'].items() if r['scope']=='FIRST'}
    heads=dict(old_heads);pairs={};files=dict(u.read(root/'SOURCE_LOCK.json')['files'])
    for seed in p['seeds']:
        for mode in ('CURRENT','DELTA'):
            final=train/f'{mode}_s{seed}'/'FINAL.pt';r=u.read(final.parent/'RESULT.json')
            if r['status']!='COMPLETE' or r['completed_steps']!=3000 or u.sha(final)!=r['final_sha256']:
                raise ValueError('NONFINAL_TRAINING')
            saved=torch.load(final,map_location='cpu',weights_only=False)
            if saved['binding']!=dict(config_sha256=u.sha(train/'TRAINING_CONFIG.json'),seed=seed,mode=mode):
                raise ValueError('HEAD_BINDING_CHANGED')
            a=f'ALIGNED_{mode}_FIRST_s{seed}'
            heads[a]=dict(path=str(final),sha256=r['final_sha256'],seed=seed,mode=mode,scope='FIRST',
                training_config_path=str(train/'TRAINING_CONFIG.json'),training_config_sha256=u.sha(train/'TRAINING_CONFIG.json'),
                training_action_scope=config['action_scope'])
            pairs[f'ALIGNED_minus_OLD_{mode}_s{seed}']=dict(CURRENT=f'OLD_{mode}_FIRST_s{seed}',DELTA=a)
            files[str(final)]=r['final_sha256']
        pairs[f'ALIGNED_DELTA_minus_CURRENT_s{seed}']=dict(CURRENT=f'ALIGNED_CURRENT_FIRST_s{seed}',DELTA=f'ALIGNED_DELTA_FIRST_s{seed}')
    p.update(id='FIRST_ALIGNED_THIRTEEN_ARM_UNSEEN_V1',heads=heads,comparisons=pairs,
        expected_head_count=12,expected_comparison_count=9,planned_groups=80,planned_executions=1040,
        primary='Matched ALIGNED minus OLD within mode and seed; actual native rerun, all FIRST scope.',
        selection='Reuse all pre-existing discovery80 in unchanged order; no selection by new score.',
        exposure='Repeatedly exposed val_unseen development block, not blind and not full1839.',
        training_change='Only gradient-bearing actor positions: FIRST instead of ALL. Unchanged class weights and coefficients.')
    run=root/'unseen';run.mkdir(exist_ok=False)
    u.write(run/'PROTOCOL.json',p);u.write(run/'DATA_MANIFEST.json',u.read(root/'EVALUATION_MANIFEST.json'))
    u.write(run/'EXECUTOR_BINDING.json',dict(path=str(HERE/'evaluate_worker.py'),sha256=u.sha(HERE/'evaluate_worker.py')))
    inherited=u.read(Path(op['discovery_protocol']).parent/'SOURCE_LOCK.json')['files'];files.update(inherited)
    for name in ['PROTOCOL.json','DATA_MANIFEST.json','EXECUTOR_BINDING.json']:files[str(run/name)]=u.sha(run/name)
    u.write(run/'SOURCE_LOCK.json',dict(files=files))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',action='store_true');a=p.parse_args()
    (evaluation if a.evaluation else training)(a.run.resolve())
