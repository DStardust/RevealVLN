"""Register the full complement of discovery80, without choosing by results."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ADAPT=HERE.parent;BASE=ADAPT.parent
sys.path.insert(0,str(BASE))
import common as u


def complement(full,discovery):
    ids={e['id'] for e in full};prior={e['id'] for e in discovery}
    if len(ids)!=len(full) or len(prior)!=len(discovery) or not prior<=ids:
        raise ValueError('DUPLICATE_OR_UNKNOWN_DISCOVERY_ID')
    return [e for e in full if e['id'] not in prior]


def prepare():
    root=HERE/'runs/confirm_001';run=root/'unseen';discovery=ADAPT/'scope_validation_v1/runs/scope_001/unseen'
    full=ADAPT/'runs/experiment_002/unseen/DATA_MANIFEST.json'
    if root.exists():raise ValueError('REGISTERED_RUN_ALREADY_EXISTS')
    if not u.read(HERE/'CPU_TEST_RESULT.json')['passed']:raise ValueError('CPU_TESTS_REQUIRED')
    if u.read(discovery/'RESULT.json')['complete_groups']!=80:raise ValueError('DISCOVERY_NOT_COMPLETE')
    u.verify_sources(discovery)
    original=u.read(full);prior=u.read(discovery/'DATA_MANIFEST.json')['episodes']
    selected=complement(original['episodes'],prior)
    if len(selected)!=289 or len(original['episodes'])!=369:raise ValueError('REMAINDER_DENOMINATOR_CHANGED')
    p=u.read(discovery/'PROTOCOL.json');heads={a:h for a,h in p['heads'].items() if h['scope']=='FIRST'}
    if len(heads)!=6:raise ValueError('MISSING_FIXED_FIRST_HEAD')
    comparisons={f'DELTA_minus_CURRENT_FIRST_s{s}':{'CURRENT':f'CURRENT_FIRST_s{s}','DELTA':f'DELTA_FIRST_s{s}'} for s in p['seeds']}
    p.update(id='FIRST_FROZEN_REMAINDER_CONFIRMATION_V1',heads=heads,comparisons=comparisons,
        planned_groups=289,planned_executions=2023,primary='Remainder289 same-process DELTA-FIRST minus CURRENT-FIRST and NATIVE',
        secondary='Discovery80 plus remainder289 descriptive pooled result, explicitly selected/exposed',
        discovery_run=str(discovery),discovery_result_sha256=u.sha(discovery/'RESULT.json'),
        selection='Exact complement of discovery80 in frozen369, original order, no score-based selection',
        exposure='Excluded from scope80 decision; prior baseline/other-method outcomes already exposed. Not blind.',
        action_scope='FIRST_ONLY_UNCHANGED_FROZEN_WEIGHTS',
        decision='Keep mean-SR development signal only with positive mean and at least 2/3 seeds positive; inspect uncertainty, SPL and costs; no automatic adoption.',
        arm_order='NATIVE then six FIRST heads rotated by immutable original episode id')
    run.mkdir(parents=True);u.write(run/'PROTOCOL.json',p)
    u.write(run/'DATA_MANIFEST.json',dict(original,episodes=selected,source_manifest_sha256=u.sha(full),scope=p['selection']))
    u.write(run/'EXECUTOR_BINDING.json',dict(path=str(HERE/'evaluate_worker.py'),sha256=u.sha(HERE/'evaluate_worker.py')))
    audit=dict(full_ids=[e['id'] for e in original['episodes']],discovery_ids=[e['id'] for e in prior],
        followup_ids=[e['id'] for e in selected],overlap=[],discovery_houses=sorted({e['house'] for e in prior}),
        followup_houses=sorted({e['house'] for e in selected}),new_training=False,
        prior_result_exposure_preserved=True,classification='FOLLOWUP_OF_EXPOSED_BENCHMARK_NOT_NEW_BLIND_SPLIT')
    u.write(root/'SPLIT_AUDIT.json',audit)
    files=dict(u.read(discovery/'SOURCE_LOCK.json')['files'])
    for f in HERE.glob('*.py'):files[str(f)]=u.sha(f)
    for f in [full,discovery/'RESULT.json',discovery/'SOURCE_LOCK.json',root/'SPLIT_AUDIT.json']:
        files[str(f)]=u.sha(f)
    for n in ['PROTOCOL.json','DATA_MANIFEST.json','EXECUTOR_BINDING.json']:files[str(run/n)]=u.sha(run/n)
    u.write(run/'SOURCE_LOCK.json',dict(files=files))
    operation=u.read(discovery.parent/'PROTOCOL.json')
    operation.update(id='FIRST_CONFIRMATION_OPERATION_V1',planned_groups=289,planned_executions=2023,
        first_complete_group_id=selected[0]['id'],phase_gpu_hour_limits=dict(TRAIN=0.,EVALUATE=40.),
        wall_hours_limit=12,chunk_groups=8,phases=['PREPARE_FROZEN_HEADS','FIRST_UNSEEN_GROUP','UNSEEN','REVIEW'],
        inherited_phase_gpu_hours=dict(TRAIN=0.,EVALUATE=0.),inherited_wall_seconds=0.,
        revision='Remove ALL arms; retain exact FIRST heads and behavior. Entire remaining289; no new optimization.',
        prior_scope_discovery_gpu_hours=u.read(discovery.parent/'RESULT.json')['gpu_hours'])
    operation.pop('wait_for_run',None);operation.pop('queue_wait_hours',None)
    u.write(root/'PROTOCOL.json',operation)
    files=dict(files);files[str(root/'PROTOCOL.json')]=u.sha(root/'PROTOCOL.json');files[str(run/'SOURCE_LOCK.json')]=u.sha(run/'SOURCE_LOCK.json')
    u.write(root/'SOURCE_LOCK.json',dict(files=files));print(str(root))


if __name__=='__main__':prepare()
