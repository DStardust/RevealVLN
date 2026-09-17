"""Actual FIT-policy replay and isolated one-step geometric teacher audit."""
import os
# Habitat EGL uses the explicit physical gpu_device_id, as in the frozen executor.
os.environ.pop('CUDA_VISIBLE_DEVICES',None)
import ast,collections,copy,hashlib,importlib.util,json,math,time
from pathlib import Path
import habitat_sim as hs
import numpy as np
import quaternion
from PIL import Image
HERE=Path(__file__).resolve().parent;OUT=HERE/'run_001'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('recovery_common',HERE/'common.py');k=load('recovery_contracts',HERE/'contracts.py')
FIT=c.LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def pose(agent):
    s=agent.get_state()
    return dict(position=s.position.tolist(),rotation=quaternion.as_float_array(s.rotation).tolist())
def same_pose(a,b):
    return k.displacement(a['position'],b['position'])<=1e-6 and k.angle(a['rotation'],b['rotation'])<1e-5
def pixels(obs):return np.ascontiguousarray(obs['rgb'][:,:,:3])
def pixel_hash(rgb):return hashlib.sha256(rgb.tobytes()).hexdigest()

def main():
    c.verify_lock();p=read(HERE/'PROTOCOL.json')
    episodes=read(FIT/'EPISODES_PRIVILEGED.json')
    source=load('recovery_sealed_reuse',FIT/'reuse.py')
    tree=ast.parse(source.source('executor.py'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_sim')
    # The simulator constructor is byte-for-byte AST reuse of the already audited source.
    env=dict(hs=hs,c=c);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(FIT/'executor.py')+':constructor','exec'),env)
    build_sim=env['build_sim']
    distance_module=load('recovery_official_distance',FIT/'official_distance.py')
    records={}
    for lane in (FIT/'run_001/lanes').iterdir():
        pp=lane/'POLICY_STEPS.jsonl'
        if not pp.exists():continue
        pg=collections.defaultdict(list);sg=collections.defaultdict(list)
        initial={r['index']:r for r in lines(lane/'INTERFACE.jsonl')}
        for row in lines(pp):pg[row['index']].append(row)
        for row in lines(lane/'STEPS_PRIVILEGED.jsonl'):sg[row['index']].append(row)
        for idx,rr in pg.items():
            assert idx not in records and len(rr)==len(sg[idx])
            records[idx]=(rr,sg[idx],initial[idx])
    assert set(records)==set(range(64)) and sum(len(v[0]) for v in records.values())==7225
    content=OUT/'content';content.mkdir()
    counts=collections.Counter();counts['planned_episodes']=64
    groups=collections.defaultdict(list);inputs={};completed=[];sim=None;scene=None;began=time.monotonic()
    def budget():
        assert time.monotonic()-began<=p['wall_seconds'],'WALL_BUDGET'
        assert counts['explicit_primitives']<=32000,'ACTION_BUDGET'
    def progress(phase,index=None,step=None):
        c.write(OUT/'PROGRESS.json',dict(status=phase,unix=time.time(),completed=len(completed),total=64,
            total_actions=counts['explicit_primitives'],episode_index=index,step=step,counts=dict(counts)))
    def render():
        rgb=pixels(sim.get_sensor_observations());counts['renders']+=1;return rgb
    def init(e):
        sim.seed(p['environment_seed'])
        state=hs.AgentState();state.position=np.array(e['start_position'],dtype=np.float32)
        state.rotation=np.quaternion(e['start_rotation'][3],*e['start_rotation'][:3])
        sim.initialize_agent(0,state);obs=sim.reset();counts['resets']+=1
        return pixels(obs)
    def geom(e,pos):
        value=distance_module.measure(sim.pathfinder,pos,[e['goals'][0]['position']],e)
        assert math.isfinite(value) and value>=0;return value
    try:
        for idx,e in enumerate(episodes):
            house=e['scene_id'].split('/')[-2]
            assert house in p['houses']
            if scene!=house:
                if sim:sim.close()
                sim=build_sim(house,p);scene=house
            pol,steps,initial=records[idx]
            rgb=init(e);history=[];recent=[];agent=sim.get_agent(0)
            follower=hs.GreedyGeodesicFollower(sim.pathfinder,agent,goal_radius=.35,fix_thrashing=False)
            for j,(decision,observed) in enumerate(zip(pol,steps)):
                budget()
                assert decision['step']==observed['step']==j+1 and decision['action']==observed['action']
                before=pose(agent);digest=pixel_hash(rgb)
                expected_hash=initial['rgb_sha256'] if j==0 else steps[j-1]['rgb_sha256']
                expected_position=e['start_position'] if j==0 else steps[j-1]['position']
                assert digest==expected_hash,'FIRST_REPLAY_RGB_MISMATCH'
                assert k.displacement(before['position'],expected_position)<=1e-6,'FIRST_REPLAY_POSE_MISMATCH'
                counts['first_replay_frames']+=1
                image=content/(digest+'.png')
                if not image.exists():
                    Image.fromarray(rgb).save(image);counts['unique_pngs']+=1
                recent=(recent+[digest])[-2:]
                value=k.payload(e['instruction']['instruction_text'],recent,history[-8:])
                ident=k.key(value);inputs.setdefault(ident,value)
                d=geom(e,before['position'])
                expected_d=read(FIT/'run_001/lanes'/f'lane_{idx%8:02d}'/f'episode_{idx:02d}.json')['start_distance_m'] if j==0 else steps[j-1]['distance_to_goal']
                assert abs(d-expected_d)<1e-5,'OFFICIAL_DISTANCE_REPLAY_MISMATCH'
                target=None;reason=None;teacher=None;branch=None
                if k.stop_at(d):
                    target=3;teacher='STOP';counts['teacher_stop_labels']+=1
                else:
                    counts['teacher_planner_queries']+=1
                    try:
                        follower.reset()
                        teacher=follower.next_action_along(np.array(e['goals'][0]['position'],dtype=np.float32))
                    except hs.errors.GreedyFollowerError:
                        reason='TEACHER_QUERY_FAILED'
                    assert same_pose(before,pose(agent)),'TEACHER_MUTATED_REAL_STATE'
                    if teacher in k.ACTIONS[:3]:
                        backup=copy.deepcopy(agent.get_state())
                        try:
                            counts['explicit_primitives']+=1;counts['teacher_verification_primitives']+=1
                            obs=sim.step(teacher)
                            after=pose(agent);after_distance=geom(e,after['position'])
                            valid=k.movement_valid(teacher,before,after,bool(obs['collided']),d,after_distance)
                            branch=dict(action=teacher,before=before,after=after,collided=bool(obs['collided']),
                                distance_before=d,distance_after=after_distance,physical_valid=valid)
                            if valid:target=k.ACTIONS.index(teacher)
                            else:reason='TEACHER_ONE_STEP_NOT_ADMISSIBLE'
                        finally:
                            agent.set_state(backup,reset_sensors=False)
                            assert same_pose(before,pose(agent)),'TEACHER_RESTORE_POSE_MISMATCH'
                            restored=render();assert pixel_hash(restored)==digest,'TEACHER_RESTORE_RGB_MISMATCH'
                            counts['counterfactual_restorations_verified']+=1
                    elif reason is None:reason='UNEXPECTED_TEACHER_STOP_OR_ACTION'
                row=dict(record_id=ident,source_index=idx,episode_id=e['episode_id'],scene_group=house,split='FIT',
                    decision_step=j,original_action=decision['action'],target=target,teacher_action=teacher,
                    distance_to_goal=d,reason=reason,teacher_branch=branch,input_pixel_hash=digest,
                    state_before=before,state_after_restore=pose(agent))
                groups[ident].append(row)
                c.append(OUT/'SUPERVISION_ONLY.jsonl',row)
                counts['visited_decisions']+=1;counts['unknown_labels']+=int(target is None)
                if target is not None:
                    counts['target_'+k.ACTIONS[target]]+=1
                    counts['teacher_student_disagreements']+=int(k.ACTIONS[target]!=decision['action'])
                if j%20==0:progress('REPLAY_AND_TEACHER_LABELING',idx,j)
                # Continue the ORIGINAL model action, never the counterfactual teacher branch.
                a=decision['action']
                if a!='STOP':
                    counts['explicit_primitives']+=1;counts['first_replay_primitives']+=1
                    obs=sim.step(a)
                    rgb=pixels(obs);history.append(a)
                    assert bool(obs['collided'])==observed['collided'],'FIRST_REPLAY_COLLISION_MISMATCH'
                else:assert j==len(pol)-1
                assert pixel_hash(rgb)==observed['rgb_sha256'],'FIRST_POST_ACTION_RGB_MISMATCH'
                assert k.displacement(pose(agent)['position'],observed['position'])<=1e-6,'FIRST_POST_ACTION_POSE_MISMATCH'
                c.append(OUT/'REPLAY_AUDIT.jsonl',dict(pass_id=1,index=idx,decision_step=j,
                    before_rgb_sha256=digest,after_rgb_sha256=pixel_hash(rgb),position=pose(agent)['position'],
                    action=a,collided=bool(obs['collided']) if a!='STOP' else False))
            # Independent second reset/replay without any teacher branch.
            rgb=init(e)
            assert pixel_hash(rgb)==initial['rgb_sha256']
            for j,(decision,observed) in enumerate(zip(pol,steps)):
                budget();replay_before_hash=pixel_hash(rgb)
                assert replay_before_hash==(initial['rgb_sha256'] if j==0 else steps[j-1]['rgb_sha256'])
                counts['second_replay_frames']+=1
                if decision['action']!='STOP':
                    counts['explicit_primitives']+=1;counts['second_replay_primitives']+=1
                    obs=sim.step(decision['action'])
                    rgb=pixels(obs);assert bool(obs['collided'])==observed['collided']
                assert pixel_hash(rgb)==observed['rgb_sha256']
                assert k.displacement(pose(agent)['position'],observed['position'])<=1e-6
                c.append(OUT/'REPLAY_AUDIT.jsonl',dict(pass_id=2,index=idx,decision_step=j,
                    before_rgb_sha256=replay_before_hash,
                    after_rgb_sha256=pixel_hash(rgb),position=pose(agent)['position'],action=decision['action'],
                    collided=bool(obs['collided']) if decision['action']!='STOP' else False))
                if j%50==0:progress('INDEPENDENT_SECOND_REPLAY',idx,j)
            completed.append(idx)
            c.append(OUT/'EPISODE_CERTIFICATES.jsonl',dict(index=idx,house=house,decisions=len(pol),
                first_replay_exact=True,second_replay_exact=True,counterfactual_restorations_exact=True))
            progress('EPISODE_CERTIFIED',idx)
        accepted=[];quarantined=[];disagreement_houses=set()
        for ident,rows in groups.items():
            if not k.admissible_group(rows):
                quarantined.append(dict(record_id=ident,occurrences=len(rows),targets=[x['target'] for x in rows]))
                continue
            first=rows[0];value=inputs[ident]
            input_row=dict(record_id=ident,**value)
            c.append(OUT/'POLICY_INPUTS.jsonl',input_row)
            accepted.append(dict(record_id=ident,target=first['target'],scene_groups=sorted({x['scene_group'] for x in rows}),
                source_occurrences=len(rows),source_indices=sorted({x['source_index'] for x in rows}),
                disagrees_with_student=any(k.ACTIONS[x['target']]!=x['original_action'] for x in rows)))
            if accepted[-1]['disagrees_with_student']:disagreement_houses.update(accepted[-1]['scene_groups'])
        c.write(OUT/'TRAINING_TARGETS.json',accepted,True)
        c.write(OUT/'QUARANTINED_GROUPS.json',quarantined,True)
        result=dict(status='COLLECTION_COMPLETE_PENDING_AUDIT',unix=time.time(),counts=dict(counts),
            completed_episodes=completed,valid_unique_advice=len(accepted),unique_input_groups=len(groups),
            quarantined_groups=len(quarantined),houses=sorted({h for x in accepted for h in x['scene_groups']}),
            unique_disagreements=sum(x['disagrees_with_student'] for x in accepted),disagreement_houses=sorted(disagreement_houses),
            target_classes=dict(collections.Counter(k.ACTIONS[x['target']] for x in accepted)),
            wall_seconds=time.monotonic()-began,navigation_gain_verified=False,training_started=False)
        c.write(OUT/'COLLECTION_RESULT.json',result,True);progress('COMPLETE_PENDING_AUDIT')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        c.write(OUT/'COLLECTION_FAILURE.json',dict(error=repr(exc),counts=dict(counts),completed=completed),True)
        progress('FAILED');raise
    finally:
        if sim:sim.close()
if __name__=='__main__':main()
