"""Freeze 80 house-stratified exposed unseen routes without reading scores."""
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ADAPT=HERE.parent;BASE=ADAPT.parent
sys.path.insert(0,str(BASE))
import common as u


def prepare():
    root=HERE/'runs/scope_001';run=root/'unseen';old=ADAPT/'runs/experiment_002'
    if root.exists():raise ValueError('DO_NOT_OVERWRITE_REGISTERED_RUN')
    if not u.read(HERE/'CPU_TEST_RESULT.json')['passed']:raise ValueError('CPU_TESTS_NOT_PASSED')
    source=old/'unseen';u.verify_sources(source);oldp=u.read(source/'PROTOCOL.json')
    manifest=u.read(source/'DATA_MANIFEST.json');houses=defaultdict(list)
    for e in manifest['episodes']:houses[e['house']].append(e)
    pools={h:deque(sorted(rows,key=lambda e:hashlib.sha256(('scope_v1:'+str(e['id'])).encode()).hexdigest())) for h,rows in houses.items()}
    selected=[]
    while len(selected)<80:
        for h in sorted(pools):
            if pools[h]:selected.append(pools[h].popleft())
            if len(selected)==80:break
        if not any(pools.values()) and len(selected)<80:raise ValueError('INSUFFICIENT_SOURCE_ROUTES')
    heads={};comparisons={}
    for seed in oldp['seeds']:
        for mode in ('CURRENT','DELTA'):
            original=oldp['heads'][f'{mode}_s{seed}']
            for scope in ('ALL','FIRST'):heads[f'{mode}_{scope}_s{seed}']=dict(original,scope=scope)
            comparisons[f'{mode}_FIRST_minus_ALL_s{seed}']={'CURRENT':f'{mode}_ALL_s{seed}','DELTA':f'{mode}_FIRST_s{seed}'}
        for scope in ('ALL','FIRST'):
            comparisons[f'DELTA_minus_CURRENT_{scope}_s{seed}']={'CURRENT':f'CURRENT_{scope}_s{seed}','DELTA':f'DELTA_{scope}_s{seed}'}
    p=dict(oldp,id='EXECUTION_SCOPE_FIXED_HEAD_DIAGNOSIS_V1',heads=heads,comparisons=comparisons,
        planned_groups=80,planned_executions=1040,optimizer_updates=0,
        action_scope='REGISTERED_FIRST_OR_ALL_WITH_SAME_FROZEN_HEAD',
        primary='FIRST minus ALL using identical trained weights, per mode/seed; SR and rescue/regression costs',
        relation_to_v4='Exposure-scope engineering diagnosis; no new training or architecture contribution',
        selection='House round-robin, SHA256(scope_v1:id) order; no reading outcomes',
        arm_order='NATIVE then 12 scope/head combinations rotated by immutable original episode id')
    run.mkdir(parents=True)
    u.write(run/'PROTOCOL.json',p)
    u.write(run/'DATA_MANIFEST.json',dict(manifest,episodes=selected,source_manifest_sha256=u.sha(source/'DATA_MANIFEST.json'),scope=p['selection']))
    u.write(run/'EXECUTOR_BINDING.json',dict(path=str(HERE/'evaluate_worker.py'),sha256=u.sha(HERE/'evaluate_worker.py')))
    files=dict(u.read(source/'SOURCE_LOCK.json')['files'])
    for path in HERE.glob('*.py'):files[str(path)]=u.sha(path)
    for name in ('PROTOCOL.json','DATA_MANIFEST.json','EXECUTOR_BINDING.json'):files[str(run/name)]=u.sha(run/name)
    u.write(run/'SOURCE_LOCK.json',dict(files=files))
    operation=dict(u.read(old/'PROTOCOL.json'),id='EXECUTION_SCOPE_OPERATION_V1',
        planned_groups=80,planned_executions=1040,first_complete_group_id=selected[0]['id'],
        inherited_phase_gpu_hours=dict(TRAIN=0.,EVALUATE=0.),inherited_wall_seconds=0.,
        phase_gpu_hour_limits=dict(TRAIN=0.,EVALUATE=24.),wall_hours_limit=12,chunk_groups=4,
        wait_for_run=str(old),queue_wait_hours=8,optimizer_updates=0,
        prior_training_gpu_hours=u.read(old/'STATUS.json')['phase_gpu_hours']['TRAIN'],
        revision='One scope change with frozen weights, 13 arms; all old metrics retained.',
        phases=['WAIT_FOR_PREDECESSOR','FIRST_UNSEEN_GROUP','UNSEEN','REVIEW'])
    u.write(root/'PROTOCOL.json',operation)
    files=dict(files);files[str(root/'PROTOCOL.json')]=u.sha(root/'PROTOCOL.json')
    files[str(run/'SOURCE_LOCK.json')]=u.sha(run/'SOURCE_LOCK.json')
    u.write(root/'SOURCE_LOCK.json',dict(files=files))
    print(str(root))


if __name__=='__main__':prepare()
