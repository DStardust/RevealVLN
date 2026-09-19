"""Privileged simulator service: only original instruction and RGB cross to policy."""
import base64
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import habitat_sim as hs
import numpy as np
import quaternion
from PIL import Image

LOCAL=Path(__file__).resolve().parent
sys.path.insert(0,str(LOCAL))
from transport import request_socket
HERE=LOCAL.parent/'ordinary_cycle_pair_recovery_v5'
def load(name):
    source=HERE/(name+'.py') if name=='common' else HERE.parent/'ordinary_cycle_pair_gpu1_v3'/(name+'.py')
    s=importlib.util.spec_from_file_location(name,source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');metric_module=c.load('official_metrics',c.TINY/'metrics.py');path_metrics=load('path_metrics');official_distance=load('official_distance')
SESSION=Path(sys.argv[2]);ARM=sys.argv[3]
OUT=None


class SimAdapter:
    def __init__(self,sim): self.sim=sim
    def get_agent_state(self): return self.sim.get_agent(0).get_state()
    def geodesic_distance(self,position,goals,episode):
        distance=official_distance.measure(self.sim.pathfinder,position,goals,episode)
        if not math.isfinite(distance):raise ValueError('UNREACHABLE_METRIC_PATH')
        return distance


def build_sim(scene,p):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_id=str(c.ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb')
    cfg.gpu_device_id=p['gpu'];cfg.enable_physics=False;cfg.allow_sliding=p['allow_sliding']
    agent=hs.agent.AgentConfiguration();agent.height=p['agent_height'];agent.radius=p['agent_radius']
    sensor=hs.SensorSpec();sensor.uuid='rgb';sensor.sensor_type=hs.SensorType.COLOR
    sensor.resolution=[224,224];sensor.position=[0,p['camera_height'],0];sensor.orientation=[0,0,0]
    sensor.parameters['hfov']=str(p['hfov']);sensor.gpu2gpu_transfer=False
    agent.sensor_specifications=[sensor]
    agent.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=value)) for name,value in
                        [('move_forward',p['forward_m']),('turn_left',p['turn_deg']),('turn_right',p['turn_deg'])]}
    sim=hs.Simulator(hs.Configuration(cfg,[agent]));assert sim.pathfinder.is_loaded
    return sim


def main():
    global OUT
    p=c.read(SESSION/'CONFIG.json')
    episodes=c.read(c.V3/'EPISODES_PRIVILEGED.json')[:100]
    order=c.read(HERE/'PAIR_ORDER.json')
    schedule=[order[r]['index'] for r in p['scheduled_ranks']]
    gt=json.load(gzip.open(p['gt_path']))
    preflight=c.read(c.V3/'GEOMETRY_PREFLIGHT.json')['rows']
    sock=request_socket(int(sys.argv[1]))
    stream=sock.makefile('rw');sim=None;scene=None;active=None;completed=0
    def send(obj):stream.write(json.dumps(obj)+'\n');stream.flush()
    def picture(rgb,index,steps):
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        path=SESSION/'frames'/(digest+'.png')
        if not path.exists():
            tmp=path.with_name(digest+'.'+ARM+'.tmp.png');Image.fromarray(rgb).save(tmp);os.replace(tmp,path)
    try:
        for line in stream:
            req=json.loads(line)
            if req=={'op':'close'}:
                assert active is None
                send(dict(closed=True));break
            reset=req.get('op')=='reset'
            if reset:
                assert set(req)=={'op','index'} and active is None and completed<len(schedule) and req['index']==schedule[completed]
                index=req['index'];e=episodes[index];h=e['scene_id'].split('/')[-2]
                OUT=SESSION/'pairs'/('pair_%03d'%p['scheduled_ranks'][completed])/ARM
                active=dict(index=index,episode_id=e['episode_id'],trajectory_id=e['trajectory_id'],house=h,
                            instruction=e['instruction']['instruction_text'],started_unix=time.time())
                c.append(OUT/'EPISODES.jsonl',dict(active,event='start'))
                if scene!=h:
                    if sim:sim.close()
                    sim=build_sim(h,p);scene=h
                sim.seed(p['environment_seed'])
                state=hs.AgentState();state.position=np.array(e['start_position'],dtype=np.float32)
                state.rotation=np.quaternion(*c.xyzw_to_wxyz(e['start_rotation']))
                sim.initialize_agent(0,state);obs=sim.reset();actual=sim.get_agent(0).get_state()
                assert np.allclose(actual.position,state.position,atol=1e-6,rtol=0)
                assert np.allclose(quaternion.as_float_array(actual.rotation),quaternion.as_float_array(state.rotation),atol=1e-6,rtol=0)
                rgb=np.ascontiguousarray(obs['rgb'][:,:,:3])
                assert rgb.shape==(224,224,3) and np.std(rgb)>0
                metrics=metric_module.OfficialMetrics(SimAdapter(sim),e['goals'][0]['position'])
                values=metrics.get();start_distance=values['distance_to_goal'];assert start_distance>0
                assert preflight[index]['episode_id']==e['episode_id']
                assert abs(start_distance-preflight[index]['official_start_distance'])<1e-5,'OFFICIAL_PREFLIGHT_DISTANCE_MISMATCH'
                positions=[actual.position.tolist()];distances=[start_distance];steps=0;collisions=0;traveled=0.;done=False
                action_counts={a:0 for a in c.ACTIONS}
                c.append(OUT/'INTERFACE.jsonl',dict(index=index,initial_pose_exact=True,physical_gpu=p['gpu'],
                    source_distance=e['info']['geodesic_distance'],measured_distance=start_distance,
                    cache_delta=start_distance-e['info']['geodesic_distance'],official_cpu_preflight_agrees=True,
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),policy_fields=['instruction','rgb','done']))
                c.write(OUT/'INITIAL_STATE.json',dict(position=actual.position.tolist(),rotation=quaternion.as_float_array(actual.rotation).tolist(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),seed=p['environment_seed'],scene=h),True)
                picture(rgb,index,0)
            else:
                assert set(req)=={'op','action'} and req['op']=='action' and active is not None
                action=req['action'];steps,done=c.advance(action,steps,done,p['max_steps'])
                action_counts[action]+=1;stopped=action=='STOP'
                if not stopped:
                    obs=sim.step(action);collided=bool(obs['collided']);collisions+=int(collided)
                    rgb=np.ascontiguousarray(obs['rgb'][:,:,:3])
                else:
                    collided=False
                current=sim.get_agent(0).get_state().position.tolist()
                traveled+=float(np.linalg.norm(np.asarray(current)-positions[-1]))
                positions.append(current);values=metrics.update(stopped);distances.append(values['distance_to_goal'])
                c.append(OUT/'STEPS_PRIVILEGED.jsonl',dict(index=index,step=steps,action=action,position=current,
                    collided=collided,distance_to_goal=values['distance_to_goal'],rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
                picture(rgb,index,steps)
                if done:
                    osr=any(v<3. for v in distances);success=bool(values['success'])
                    failure='success' if success else ('entered_goal_but_no_successful_stop' if osr else
                            'stopped_without_reaching_goal' if stopped else 'budget_exhausted_without_reaching_goal')
                    ndtw=path_metrics.ndtw(positions,gt[str(e['episode_id'])]['locations'])
                    record=dict(active,event='complete',steps=steps,stopped=stopped,termination='STOP' if stopped else 'MAX_STEPS',
                        success=values['success'],spl=values['spl'],navigation_error_m=values['distance_to_goal'],
                        oracle_success=float(osr),min_distance_m=min(distances),start_distance_m=start_distance,
                        path_length_m=traveled,collisions=collisions,action_counts=action_counts,
                        wall_seconds=time.time()-active['started_unix'],failure_category=failure,
                        service_failure=False,ndtw=ndtw,sdtw=ndtw*float(success),ndtw_backend='exact_Euclidean_DTW_FDTW_false')
                    c.append(OUT/'EPISODES.jsonl',record)
                    c.write(OUT/f'episode_{index:02d}.json',dict(record,positions=positions,distances=distances),True)
                    active=None;completed+=1;send({'done':True});continue
            response=dict(done=False,rgb=base64.b64encode(rgb.tobytes()).decode())
            if reset:response['instruction']=e['instruction']['instruction_text']
            send(response)
    except BaseException as exc:
        c.append((OUT or SESSION)/'SERVICE_FAILURES.jsonl',dict(active or {},error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if sim:sim.close()
        stream.close();sock.close()

if __name__=='__main__':main()


