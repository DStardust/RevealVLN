"""Independent label/causal/replay audit; rejects unknown or conflicting groups."""
import collections,importlib.util,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'run_001'
def load(name):
    s=importlib.util.spec_from_file_location('audit_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');k=load('contracts')
FIT=c.LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
    c.verify_lock();r=read(OUT/'COLLECTION_RESULT.json');launch=read(OUT/'LAUNCH_RESULT.json')
    assert r['status']=='COLLECTION_COMPLETE_PENDING_AUDIT'
    assert launch['status']=='COMPLETE' and not launch['cleanup']['remaining']
    assert launch['foreign_processes_signaled']==[] and launch['optimizer_updates']==0
    assert r['completed_episodes']==list(range(64))
    assert r['counts']['visited_decisions']==r['counts']['first_replay_frames']==r['counts']['second_replay_frames']==7225
    assert r['counts']['explicit_primitives']<=32000
    episodes=read(FIT/'EPISODES_PRIVILEGED.json');expected={};source_steps={};grouped=collections.defaultdict(list)
    for lane in (FIT/'run_001/lanes').iterdir():
        pp=lane/'POLICY_STEPS.jsonl'
        if not pp.exists():continue
        pg=collections.defaultdict(list);sg=collections.defaultdict(list)
        initial={x['index']:x for x in lines(lane/'INTERFACE.jsonl')}
        for x in lines(pp):pg[x['index']].append(x)
        for x in lines(lane/'STEPS_PRIVILEGED.jsonl'):sg[x['index']].append(x)
        for idx,pol in pg.items():
            steps=sg[idx];history=[];recent=[];first=read(lane/f'episode_{idx:02d}.json')['start_distance_m']
            for j,record in enumerate(pol):
                digest=initial[idx]['rgb_sha256'] if j==0 else steps[j-1]['rgb_sha256']
                recent=(recent+[digest])[-2:]
                value=k.payload(episodes[idx]['instruction']['instruction_text'],recent,history[-8:])
                d=first if j==0 else steps[j-1]['distance_to_goal']
                expected[idx,j]=dict(payload=value,key=k.key(value),distance=d,action=record['action'])
                source_steps[idx,j]=steps[j]
                if record['action']!='STOP':history.append(record['action'])
    assert len(expected)==7225
    rows=lines(OUT/'SUPERVISION_ONLY.jsonl');assert len(rows)==7225
    seen=set()
    for x in rows:
        tag=x['source_index'],x['decision_step'];assert tag not in seen;seen.add(tag);e=expected[tag]
        assert x['record_id']==e['key'] and x['input_pixel_hash']==e['payload']['rgb_sha256'][-1]
        assert x['original_action']==e['action'] and abs(x['distance_to_goal']-e['distance'])<1e-5
        assert x['split']=='FIT' and x['scene_group'] in read(HERE/'PROTOCOL.json')['houses']
        assert k.displacement(x['state_before']['position'],x['state_after_restore']['position'])<=1e-6
        assert k.angle(x['state_before']['rotation'],x['state_after_restore']['rotation'])<1e-5
        if k.stop_at(e['distance']):
            assert x['target']==3 and x['teacher_action']=='STOP' and x['teacher_branch'] is None
        elif x['target'] is not None:
            assert x['target'] in (0,1,2) and x['teacher_action']==k.ACTIONS[x['target']]
            b=x['teacher_branch'];assert b['before']==x['state_before']
            assert k.movement_valid(b['action'],b['before'],b['after'],b['collided'],b['distance_before'],b['distance_after'])
            assert b['physical_valid'] and abs(b['distance_before']-e['distance'])<1e-5
        else:assert x['reason']
        grouped[x['record_id']].append(x)
    replay=lines(OUT/'REPLAY_AUDIT.jsonl');assert len(replay)==14450
    audited=set()
    for x in replay:
        tag=x['index'],x['decision_step'];e=expected[tag];s=source_steps[tag]
        triple=x['pass_id'],*tag;assert triple not in audited and x['pass_id'] in (1,2);audited.add(triple)
        assert x['before_rgb_sha256']==e['payload']['rgb_sha256'][-1] and x['after_rgb_sha256']==s['rgb_sha256']
        assert x['action']==e['action'] and x['collided']==s['collided']
        assert k.displacement(x['position'],s['position'])<=1e-6
    accepted=read(OUT/'TRAINING_TARGETS.json')
    should={key for key,rr in grouped.items() if k.admissible_group(rr)}
    assert {x['record_id'] for x in accepted}==should and len(accepted)==len(should)
    payloads=lines(OUT/'POLICY_INPUTS.jsonl')
    assert len(payloads)==len(accepted) and {x['record_id'] for x in payloads}==should
    exp_by_key={x['key']:x['payload'] for x in expected.values()}
    for x in payloads:
        assert set(x)=={'record_id','instruction','rgb_sha256','executed_actions'}
        v={f:x[f] for f in ('instruction','rgb_sha256','executed_actions')}
        assert k.key(v)==x['record_id'] and v==exp_by_key[x['record_id']]
        assert all((OUT/'content'/(digest+'.png')).is_file() for digest in x['rgb_sha256'])
    for x in accepted:
        rr=grouped[x['record_id']]
        assert x['target']==rr[0]['target']
        assert x['scene_groups']==sorted({y['scene_group'] for y in rr})
        assert x['disagrees_with_student']==any(k.ACTIONS[y['target']]!=y['original_action'] for y in rr)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    subprocess.run([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(HERE/'audit_pixels.py')],
        cwd=c.ROOT,env=env,check=True,timeout=600)
    houses={h for x in accepted for h in x['scene_groups']}
    disagreements=[x for x in accepted if x['disagrees_with_student']]
    disagree_houses={h for x in disagreements for h in x['scene_groups']}
    passed=len(accepted)>=1000 and len(houses)>=8 and len(disagreements)>=100 and len(disagree_houses)>=8
    counts=collections.Counter(k.ACTIONS[x['target']] for x in accepted)
    files=[OUT/'POLICY_INPUTS.jsonl',OUT/'TRAINING_TARGETS.json',OUT/'SUPERVISION_ONLY.jsonl',
           OUT/'REPLAY_AUDIT.jsonl',OUT/'PIXEL_AUDIT.json',OUT/'COLLECTION_RESULT.json',OUT/'QUARANTINED_GROUPS.json']
    c.write(OUT/'DATA_SEAL.json',dict(files={str(f):c.sha(f) for f in files}),True)
    result=dict(status='AUDITED_ONPOLICY_TEACHER_DATA_READY' if passed else 'DATA_GATE_NOT_MET',unix=time.time(),
        data_gate=passed,valid_unique_advice=len(accepted),houses=sorted(houses),
        unique_disagreements=len(disagreements),disagreement_houses=sorted(disagree_houses),
        target_classes=dict(counts),visited_decisions=7225,replay_frames_audited=14450,
        exact_causal_input_reconstruction=True,all_pngs_decoded=True,unknown_and_conflicts_quarantined=True,
        teacher_one_step_only_not_full_instruction_semantics=True,source_lock_verified=True,
        primitive_actions=r['counts']['explicit_primitives'],teacher_planner_queries=r['counts']['teacher_planner_queries'],
        original_strict_dataset_unchanged=True,training_started=False,navigation_gain_verified=False)
    c.write(OUT/'RESULT.json',result,True);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
