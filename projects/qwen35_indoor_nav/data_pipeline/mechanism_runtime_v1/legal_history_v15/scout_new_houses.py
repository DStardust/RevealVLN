"""Collect actual outbound/return components in frozen new houses, one GPU."""
import math
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LINE, RUNTIME, c, load, pose_delta
from content_store import AuditedContentStore


def main(run):
    config=c.read(run/'CONFIG.json')
    for path,digest in config['source_hashes'].items():assert c.sha(LINE/path)==digest,path
    backend_module=load('v15_new_scene_backend',RUNTIME/'habitat_backend.py')
    feedback=load('v15_new_scene_feedback',RUNTIME/'feedback_generation_v1/feedback.py')
    compiler_module=load('v15_new_scene_compiler',LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    factory=load('v15_new_scene_factory',LINE/'data_pipeline/mechanism_factory_v2/factory.py')
    backend=None;decisions=0;attempted_actions=0;rows=[];context={};folder=None;records=[]
    def emit(kind,value):
        nonlocal decisions
        if kind=='action_completed':
            decisions+=1
            c.append(run/'ACTIONS.jsonl',dict(decision=decisions,context=context,**value))
            if decisions%20==0:c.write(run/'PROGRESS.json',dict(unix=time.time(),decisions=decisions,context=context))
        elif kind=='trace':
            path=folder/f'trace_{len(records):04d}.json';c.write(path,value,True)
            clean=compiler.complete(value)
            seen=sorted({role for e in compiler.atoms(value['observations']) for role,ids in e.items() if ids}) if clean else []
            delta=pose_delta(value['observations'][0]['pose'],value['observations'][-1]['pose']) if value['observations'] else None
            record=dict(context=dict(context),path=str(path.relative_to(LINE)),sha256=c.sha(path),
                actions=len(value['actions']),complete=clean,seen_roles=seen,
                closed_within_1e_5=clean and delta<=1e-5,max_pose_delta=delta,
                exact_closed=clean and value['observations'][0]['pose']==value['observations'][-1]['pose'])
            records.append(record);c.append(folder/'RECORDS.jsonl',record)
    class Budget:
        def check_time(self):
            if attempted_actions>=config['max_decisions']:raise RuntimeError('DATA_DECISION_BUDGET')
        def reserve_action(self):
            nonlocal attempted_actions
            self.check_time();attempted_actions+=1
    budget=Budget()
    class Runner:
        def __init__(self,b,comp):
            self.backend=b;self.compiler=comp;self.budget=budget;self.emit=emit
        def observe(self,step):return dict(self.backend.observe(),step=step)
        def run(self,position,actions):
            self.backend.reset(position,0,config['environment_seed'])
            trace=dict(actions=[],observations=[self.observe(0)],complete=False,collisions=0,
                       initial_position=position,initial_yaw_bin=0,interior_state_assignments=0)
            try:
                for a in actions:
                    budget.reserve_action();collision=self.backend.step(a)
                    trace['actions'].append(a);trace['collisions']+=int(collision)
                    emit('action_completed',dict(action=a,collision=collision))
                    trace['observations'].append(self.observe(len(trace['actions'])))
                    if collision:break
                trace['complete']=len(trace['actions'])==len(actions)
                return trace
            finally:emit('trace',trace)
    def compact(actions):
        inverse=[]
        for a in reversed(actions):
            fragment=['L']*12+['F']+['R']*12 if a=='F' else ['R' if a=='L' else 'L']
            inverse=factory.compress(inverse+fragment)
        return actions+inverse
    try:
        with AuditedContentStore(run/'content',config['output_gib']*2**30) as store:
            for house in config['houses']:
                for path,digest in house['assets'].items():assert c.sha(path)==digest,path
                folder=run/house['house_id'];folder.mkdir();records=[]
                backend=backend_module.HabitatBackend(house['scene'],config['gpu'],house['roles'],store,
                    dict(runtime_allowed=True,scene_glb=house['scene'],gpu_device=config['gpu']))
                assert backend.eligible==house['expected_eligible']
                roles={k:[v['mpcat40'],v['room']] for k,v in house['roles'].items()};first=next(iter(roles))
                compiler=compiler_module.Compiler(roles,{'probe':dict(anchor=first,terminal=first,instruction='Offline visible-event component collection')},backend.eligible)
                c.write(folder/'INVENTORY.json',dict(objects=backend.objects,roles=house['roles'],eligible=backend.eligible),True)
                backend.sim.pathfinder.seed(config['geometry_seed'])
                proposals=[]
                for index in range(config['random_hub_candidates']):
                    position=backend.sim.pathfinder.get_random_navigable_point().tolist()
                    assert all(math.isfinite(x) for x in position)
                    groups=[]
                    for role in roles:
                        targets=feedback.candidate_targets(backend,position,role,limit=1)
                        if targets:groups.append(dict(role=role,target=targets[0]))
                    groups.sort(key=lambda r:(r['target']['distance'],r['role']))
                    proposals.append(dict(index=index,position=position,groups=groups))
                proposals.sort(key=lambda h:(-len({roles[g['role']][1] for g in h['groups']}),-len(h['groups']),h['index']))
                hubs=[]
                for item in proposals:
                    if all(math.dist(item['position'],h['position'])>=1 for h in hubs):hubs.append(item)
                    if len(hubs)==config['hubs_per_house']:break
                c.write(folder/'FROZEN_HUBS.json',dict(proposals=proposals,selected=hubs,actions_before_freeze=0),True)
                runner=Runner(backend,compiler);follower=feedback.FeedbackRunner(runner)
                for hub in hubs:
                    context=dict(house=house['house_id'],partition=house['partition'],hub=hub['index'],phase='public_tail')
                    runner.run(hub['position'],['L','R']*4)
                    for group in hub['groups'][:config['groups_per_hub']]:
                        context=dict(context,phase='outbound',role=group['role'])
                        target=group['target']
                        trace=follower.navigate(hub['position'],0,target['target'],target['center'],
                            max_actions=config['max_outbound_actions'],seed=config['environment_seed'])
                        if not compiler.complete(trace):continue
                        actions=compact(trace['actions'])
                        if len(actions)>config['max_loop_actions']:continue
                        context=dict(context,phase='closed_return')
                        runner.run(hub['position'],actions)
                summary=dict(house_id=house['house_id'],partition=house['partition'],records=records,
                    complete_traces=sum(r['complete'] for r in records),
                    closed_return_components=sum(r['complete'] and r['context']['phase']=='closed_return' and r['closed_within_1e_5'] for r in records),
                    backend_counts=backend.counts,training_admission=False)
                c.write(folder/'RESULT.json',summary,True);rows.append(summary)
                backend.close();backend=None
            store._check_root();audit=store.full_audit();assert audit['audit_pass']
            c.write(run/'CONTENT_AUDIT.json',audit,True);c.write(run/'TIMESTAMP_CHECKS.json',store.timestamp_checks,True)
        c.write(run/'RESULT.json',dict(status='NEW_HOUSE_COMPONENTS_COLLECTED',houses=rows,
            actual_decisions=decisions,attempted_actions=attempted_actions,model_loads=0,optimizer_updates=0,
            certified_families=0,training_admission=False),True)
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as error:
        c.write(run/'FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc()),True)
        raise
