"""Registered physical episodes; private teacher is enabled only during FIT collection."""
import base64,hashlib,json,math,os,socket,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
rgb_check=u.load("terminal18_rgb_r3",Path(__file__).resolve().parent/"rgb.py")
import habitat_sim as hs
import numpy as np
import quaternion
from PIL import Image
from types import SimpleNamespace

source=u.ASSET/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/executor.py'
old=u.load('terminal18_frozen_sim',source)
route=u.load('terminal18_route',u.LINE/'data_pipeline/ordinary_route_teacher_v11/route.py')
session=Path(sys.argv[2]);arm=sys.argv[3];p=u.read(session/'CONFIG.json')

def main():
    episodes=u.read(Path(p['episodes_path']));order=u.read(Path(p['order_path']));geometry=u.read(Path(p['geometry_path']))
    gt=u.groundtruth(p['gt_path']);sock=socket.socket(fileno=int(sys.argv[1]));sock.settimeout(300);stream=sock.makefile('rw')
    sim=None;scene=None;active=None;completed=0;folder=session
    def send(x):stream.write(json.dumps(x)+'\n');stream.flush()
    def picture(rgb):
        digest=hashlib.sha256(rgb.tobytes()).hexdigest();out=session/'frames'/(digest+'.png')
        if not out.exists():
            temp=out.with_name(digest+'.'+arm+'.tmp.png');Image.fromarray(rgb).save(temp);os.replace(temp,out)
        return digest
    def finish(reason):
        nonlocal active,completed
        osr=any(d<3 for d in distances);success=bool(values['success']);failure='success' if success else ('entered_goal_but_no_successful_stop' if osr else 'stopped_without_reaching_goal' if stopped else 'budget_exhausted_without_reaching_goal')
        ndtw=old.path_metrics.ndtw(positions,gt[str(e['episode_id'])]['locations'])
        record=dict(active,event='complete',steps=steps,stopped=stopped,termination=reason,success=values['success'],spl=values['spl'],navigation_error_m=values['distance_to_goal'],oracle_success=float(osr),min_distance_m=min(distances),start_distance_m=distances[0],path_length_m=traveled,collisions=collisions,action_counts=counts,wall_seconds=time.time()-active['started_unix'],failure_category=failure,service_failure=False,ndtw=ndtw,sdtw=ndtw*float(success))
        u.write(folder/f'episode_{index:02d}.json',dict(record,positions=positions,distances=distances),True);u.append(folder/'EPISODES.jsonl',record);active=None;completed+=1
    try:
        for line in stream:
            req=json.loads(line)
            if req=={'op':'close'}:
                assert active is None;send(dict(closed=True));break
            if req.get('op')=='reset':
                rank=p['scheduled_ranks'][completed];row=order[rank]
                assert set(req)=={'op','index'} and active is None and req['index']==row['index'],'REGISTERED_RESET'
                index=row['index'];e=episodes[index];house=e['scene_id'].split('/')[-2];mode=row.get('mode','POLICY')
                assert mode=='POLICY' or (p['phase']=='COLLECT' and house in p['training_houses_list']),'PRIVILEGED_TEACHER_FORBIDDEN'
                folder=session/'pairs'/f'pair_{rank:03d}'/arm
                active=dict(index=index,rank=rank,episode_id=e['episode_id'],trajectory_id=e['trajectory_id'],house=house,instruction=e['instruction']['instruction_text'],mode=mode,started_unix=time.time())
                u.append(folder/'EPISODES.jsonl',dict(active,event='start'))
                if scene!=house:
                    if sim:sim.close()
                    sim=old.build_sim(house,p);scene=house
                sim.seed(p['environment_seed']);state=hs.AgentState();state.position=np.asarray(e['start_position'],dtype=np.float32);state.rotation=np.quaternion(*u.c.xyzw_to_wxyz(e['start_rotation']))
                sim.initialize_agent(0,state);obs=sim.reset();actual=sim.get_agent(0).get_state()
                assert np.allclose(actual.position,state.position,atol=1e-6,rtol=0) and np.allclose(quaternion.as_float_array(actual.rotation),quaternion.as_float_array(state.rotation),atol=1e-6,rtol=0),'INITIAL_POSE'
                rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]);rgb_info=rgb_check.validate(rgb)
                if rgb_info['uniform']:
                    repeated=np.ascontiguousarray(sim.get_sensor_observations()['rgb'][:,:,:3]);assert np.array_equal(rgb,repeated),'UNSTABLE_UNIFORM_RGB'
                    check=sim.get_agent(0).get_state();assert np.array_equal(check.position,actual.position) and np.array_equal(quaternion.as_float_array(check.rotation),quaternion.as_float_array(actual.rotation)),'RGB_CHECK_CHANGED_POSE'
                u.write(folder/'RGB_INITIAL_AUDIT.json',dict(rgb_info,uniform_repeat_checked=rgb_info['uniform'],camera_transform=np.asarray(sim.get_agent(0)._sensors['rgb'].node.absolute_transformation()).tolist(),registered_pose_unchanged=True),True)
                metrics=old.metric_module.OfficialMetrics(old.SimAdapter(sim),e['goals'][0]['position']);values=metrics.get()
                assert geometry[index]['episode_id']==e['episode_id'] and abs(values['distance_to_goal']-geometry[index]['distance'])<1e-5,'GEOMETRY_PREFLIGHT_MISMATCH'
                assert values['distance_to_goal']>0
                steps=0;stopped=False;collisions=0;traveled=0.;counts={a:0 for a in u.c.ACTIONS};positions=[actual.position.tolist()];distances=[values['distance_to_goal']];cursor=0;meters={}
                targets=e['reference_path']+[e['goals'][0]['position']]
                follower=hs.GreedyGeodesicFollower(sim.pathfinder,sim.get_agent(0),goal_radius=.35,fix_thrashing=False) if mode=='TEACHER' else None
                digest=picture(rgb)
                u.write(folder/'INITIAL_STATE.json',dict(position=actual.position.tolist(),rotation=quaternion.as_float_array(actual.rotation).tolist(),rgb_sha256=digest,seed=p['environment_seed'],scene=house),True)
                u.append(folder/'INTERFACE.jsonl',dict(index=index,initial_pose_exact=True,physical_gpu=p['gpu'],source_distance=e['info']['geodesic_distance'],measured_distance=values['distance_to_goal'],official_cpu_preflight_agrees=True,rgb_sha256=digest,policy_fields=['instruction','rgb','done','executed_action']))
                send(dict(done=False,instruction=active['instruction'],rgb=base64.b64encode(rgb.tobytes()).decode()));continue
            assert set(req)=={'op','action'} and req['op']=='action' and active is not None
            proposed=req['action'];assert proposed in u.c.ACTIONS
            before=sim.get_agent(0).get_state();before_pos=before.position.copy();before_rot=quaternion.as_float_array(before.rotation).copy()
            action=proposed
            if mode=='TEACHER':
                def distance(target):
                    key=tuple(target)
                    if key not in meters:meters[key]=SimpleNamespace(_shortest_path_cache=None)
                    return old.official_distance.measure(sim.pathfinder,before_pos,[list(target)],meters[key])
                rt=route.advance(targets,cursor,distance);cursor=rt['cursor_after']
                if route.stop_at(rt,values['distance_to_goal']):action='STOP'
                else:
                    try:
                        follower.reset();action=follower.next_action_along(np.asarray(targets[rt['selected_index']],dtype=np.float32))
                    except hs.errors.GreedyFollowerError:action=None
                    after=sim.get_agent(0).get_state()
                    assert np.array_equal(before_pos,after.position) and np.array_equal(before_rot,quaternion.as_float_array(after.rotation)),'TEACHER_MUTATED_STATE'
                    if action not in u.c.ACTIONS[:3]:
                        finish('TEACHER_UNAVAILABLE');send(dict(done=True,executed_action=None));continue
            before_distance=values['distance_to_goal'];steps,done=u.c.advance(action,steps,False,500);stopped=action=='STOP';counts[action]+=1
            if not stopped:
                obs=sim.step(action);collided=bool(obs['collided']);rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]);collisions+=int(collided)
            else:collided=False
            current=sim.get_agent(0).get_state().position.tolist();traveled+=float(np.linalg.norm(np.asarray(current)-positions[-1]));positions.append(current);values=metrics.update(stopped);distances.append(values['distance_to_goal']);digest=picture(rgb)
            u.append(folder/'STEPS_PRIVILEGED.jsonl',dict(index=index,step=steps,action=action,position=current,collided=collided,distance_to_goal=values['distance_to_goal'],rgb_sha256=digest))
            if p['phase']=='COLLECT':u.append(folder/'SUPERVISION_ONLY.jsonl',dict(step=steps,house=house,episode_id=e['episode_id'],trajectory_id=e['trajectory_id'],mode=mode,target=int(before_distance<3),distance_before=before_distance,executed_action=action,proposed_action=proposed))
            if done:
                finish('STOP' if stopped else 'MAX_STEPS');send(dict(done=True,executed_action=action));continue
            send(dict(done=False,executed_action=action,rgb=base64.b64encode(rgb.tobytes()).decode()))
    except BaseException as exc:
        u.append(folder/'SERVICE_FAILURES.jsonl',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        if sim:sim.close()
        stream.close();sock.close()
if __name__=='__main__':main()
