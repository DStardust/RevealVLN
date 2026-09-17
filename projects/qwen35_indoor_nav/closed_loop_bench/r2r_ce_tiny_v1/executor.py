"""Privileged simulator service: only original instruction and RGB cross to policy."""
import base64
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

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');metric_module=load('metrics')
OUT=HERE/'run_001'


class SimAdapter:
    def __init__(self,sim): self.sim=sim
    def get_agent_state(self): return self.sim.get_agent(0).get_state()
    def geodesic_distance(self,position,goals,episode):
        assert len(goals)==1
        path=hs.ShortestPath();path.requested_start=np.asarray(position,dtype=np.float32)
        path.requested_end=np.asarray(goals[0],dtype=np.float32)
        if not self.sim.pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance):
            raise ValueError('UNREACHABLE_METRIC_PATH')
        return float(path.geodesic_distance)


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
    c.verify_lock()
    p=json.loads((HERE/'PROTOCOL.json').read_text())
    episodes=json.loads((HERE/'EPISODES_PRIVILEGED.json').read_text())
    sock=socket.socket(fileno=int(sys.argv[1]));sock.settimeout(180)
    stream=sock.makefile('rw');sim=None;scene=None;active=None;completed=0
    def send(obj):stream.write(json.dumps(obj)+'\n');stream.flush()
    def picture(rgb,index,steps):
        path=OUT/'frames'/f'ep_{index:02d}_step_{steps:03d}.png'
        Image.fromarray(rgb).save(path)
    try:
        for line in stream:
            req=json.loads(line)
            if req=={'op':'close'}:
                assert active is None
                send(dict(closed=True));break
            reset=req.get('op')=='reset'
            if reset:
                assert set(req)=={'op','index'} and active is None and req['index']==completed and completed<8
                index=req['index'];e=episodes[index];h=e['scene_id'].split('/')[-2]
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
                assert abs(start_distance-e['info']['geodesic_distance'])<.05,'DATASET_GEODESIC_MISMATCH'
                positions=[actual.position.tolist()];distances=[start_distance];steps=0;collisions=0;traveled=0.;done=False
                action_counts={a:0 for a in c.ACTIONS}
                c.append(OUT/'INTERFACE.jsonl',dict(index=index,initial_pose_exact=True,physical_gpu=p['gpu'],
                    source_distance=e['info']['geodesic_distance'],measured_distance=start_distance,
                    rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),policy_fields=['instruction','rgb','done']))
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
                if steps%20==0 or done:picture(rgb,index,steps)
                if done:
                    osr=any(v<3. for v in distances);success=bool(values['success'])
                    failure='success' if success else ('entered_goal_but_no_successful_stop' if osr else
                            'stopped_without_reaching_goal' if stopped else 'budget_exhausted_without_reaching_goal')
                    record=dict(active,event='complete',steps=steps,stopped=stopped,termination='STOP' if stopped else 'MAX_STEPS',
                        success=values['success'],spl=values['spl'],navigation_error_m=values['distance_to_goal'],
                        oracle_success=float(osr),min_distance_m=min(distances),start_distance_m=start_distance,
                        path_length_m=traveled,collisions=collisions,action_counts=action_counts,
                        wall_seconds=time.time()-active['started_unix'],failure_category=failure,
                        service_failure=False,ndtw=None,sdtw=None)
                    c.append(OUT/'EPISODES.jsonl',record)
                    c.write(OUT/f'episode_{index:02d}.json',dict(record,positions=positions,distances=distances),True)
                    active=None;completed+=1;send({'done':True});continue
            response=dict(done=False,rgb=base64.b64encode(rgb.tobytes()).decode())
            if reset:response['instruction']=e['instruction']['instruction_text']
            send(response)
    except BaseException as exc:
        c.append(OUT/'SERVICE_FAILURES.jsonl',dict(active or {},error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if sim:sim.close()
        stream.close();sock.close()

if __name__=='__main__':main()
