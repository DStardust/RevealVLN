"""Actually replay each registered FIT prefix and execute its earlier STOP."""
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from continuation_service import ContentStore,NoInteriorJoin,exact_pose
from evaluator_v16 import legacy,evaluate

def validate_prefix(actual,reference,stop_step):
    if actual['actions']!=reference['actions'][:stop_step]+['S']:raise ValueError('TEACHER_ACTION_PREFIX')
    if len(actual['observations'])!=stop_step+1:raise ValueError('STOP_MUST_NOT_CREATE_OBSERVATION')
    for t,(a,b) in enumerate(zip(actual['observations'],reference['observations'])):
        if a['rgb_hash']!=b['rgb_hash'] or a['semantic_hash']!=b['semantic_hash'] or not exact_pose(a['pose'],b['pose']):
            raise ValueError('PHYSICAL_PREFIX_MISMATCH:'+str(t))

def main(run):
    cfg=runtime_config(run);verify_lock(read(run/'SOURCE_LOCK.json'))
    plans=read(run/'TEACHER_PLAN.json')['plans'];assigned=set(read(Path(os.environ['B2_DEVICE']))['collection_indices'])
    families={f['family_id']:f for f in read(run/'DATA.json')['raw_families']}
    directory=run/'collect'/f"gpu_{cfg['gpu']}";directory.mkdir(parents=True,exist_ok=True)
    store=ContentStore(directory/'content',cfg['artifact_gib']*2**30);backend=None;current=None;completed=0
    began=time.monotonic()
    try:
        for rank,plan in enumerate(plans):
            if rank not in assigned:continue
            dest=run/'certificates'/plan['id'];dest.mkdir(parents=True,exist_ok=True)
            if (dest/'CERTIFICATE.json').exists():
                cert=read(dest/'CERTIFICATE.json')
                if cert['plan_sha256']!=digest(plan) or sha(LINE/cert['trace']['path'])!=cert['trace']['sha256']:raise ValueError('SEALED_TEACHER_CHANGED')
                completed+=1;continue
            attempt=dest/f'attempt_{len(list(dest.glob("attempt_*")))+1:03d}';attempt.mkdir()
            append(directory/'ATTEMPTS.jsonl',dict(plan=plan['id'],attempt=str(attempt.relative_to(LINE)),event='START',unix=time.time()))
            family=families[plan['family_id']]
            if family['split']!='FIT':raise ValueError('NONFIT_COLLECTION')
            if current!=family['family_id']:
                if backend:backend.close()
                backend=NoInteriorJoin(family['scene'],cfg['gpu'],family['roles'],store,
                    dict(runtime_allowed=True,scene_glb=family['scene'],gpu_device=cfg['gpu']))
                current=family['family_id']
            if backend.eligible!=family['compiler']['eligible']:raise ValueError('ROLE_IDENTITY_CHANGED')
            ref=plan['source_trace'];path=LINE/ref['path']
            if sha(path)!=ref['sha256']:raise ValueError('SOURCE_TRACE_CHANGED')
            reference=read(path);backend.reset(family['initial_position'],family['initial_yaw'],0)
            trace=dict(actions=[],observations=[dict(backend.observe(),step=0)],collisions=0,
                       complete=False,interior_state_assignments=0)
            for action in plan['actions']:
                if time.monotonic()-began>cfg['max_session_hours']*3600:raise TimeoutError('COLLECTION_BUDGET')
                trace['actions'].append(action)
                if action!='S':
                    trace['collisions']+=int(backend.step(action))
                    trace['observations'].append(dict(backend.observe(),step=len(trace['actions'])))
            trace['complete']=True;write(attempt/'TRACE.json',trace,True)
            if backend.counts['explicit_reconstructions']:raise ValueError('INTERIOR_JOIN')
            validate_prefix(trace,reference,plan['stop_step'])
            if any(not o['evidence_complete'] for o in trace['observations']):raise ValueError('INCOMPLETE_EVIDENCE')
            compiler=legacy.Compiler(**family['compiler']);results=[]
            for item in plan['tasks']:
                result=evaluate(compiler,trace,item['task'],item['cutoff'])
                if result['safe_v16_label']!='PASS':raise ValueError('ALTERNATIVE_TEACHER_NOT_PASS')
                results.append(dict(item,**result))
            certificate=dict(plan_sha256=digest(plan),trace=dict(path=str((attempt/'TRACE.json').relative_to(LINE)),sha256=sha(attempt/'TRACE.json')),
                content_root=str(store.root.relative_to(LINE)),tasks=results,source_prefix_exact=True,
                stop_observation_added=False,source_training_admission=family.get('training_admission'),
                admission='FIT_ONLY_CERTIFIED_ACTION_ALTERNATIVE',new_house_evidence=False)
            write(dest/'CERTIFICATE.json',certificate,True);completed+=1
            append(directory/'ATTEMPTS.jsonl',dict(plan=plan['id'],event='CERTIFIED',trace=certificate['trace'],unix=time.time()))
            write(directory/'PROGRESS.json',dict(complete=completed,planned=len(assigned),seconds=time.monotonic()-began))
            print(f'certified {completed}/{len(assigned)} plan={plan["id"]}',flush=True)
        if not (directory/'RESULT.json').exists():
            immutable(directory/'RESULT.json',dict(complete=completed,planned=len(assigned),base_loaded=False,base_updates=0,seconds=time.monotonic()-began))
        elif read(directory/'RESULT.json')['complete']!=completed:raise ValueError('COLLECTION_COMPLETION_CHANGED')
    finally:
        if backend:backend.close()

if __name__=='__main__':main(Path(sys.argv[1]))
