"""Privileged executor; only RGB bytes cross into the policy worker."""
import base64
import hashlib
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

OUT=Path(__file__).resolve().parent; LINE=OUT.parents[2]; ROOT=LINE.parents[1]
DATA=LINE/'data_pipeline/ordinary_pilot_v1'

def append(name,obj):
    with (OUT/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')

def simulator(scene):
    cfg=hs.SimulatorConfiguration()
    cfg.scene_id=str(ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb')
    cfg.gpu_device_id=int(os.environ['SFT_PHYSICAL_GPU']);cfg.enable_physics=False;cfg.allow_sliding=False
    ac=hs.agent.AgentConfiguration();ac.height=1.5;ac.radius=.1
    sensor=hs.SensorSpec();sensor.uuid='rgb';sensor.sensor_type=hs.SensorType.COLOR
    sensor.resolution=[224,224];sensor.position=[0,1.25,0];sensor.orientation=[0,0,0]
    sensor.parameters['hfov']='90';sensor.gpu2gpu_transfer=False
    ac.sensor_specifications=[sensor]
    ac.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=value)) for name,value in [('move_forward',.25),('turn_left',15.),('turn_right',15.)]}
    sim=hs.Simulator(hs.Configuration(cfg,[ac]));assert sim.pathfinder.is_loaded
    return sim

def distance(sim,p,q):
    sp=hs.ShortestPath();sp.requested_start=np.asarray(p,dtype=np.float32);sp.requested_end=np.asarray(q,dtype=np.float32)
    if not sim.pathfinder.find_path(sp) or not math.isfinite(sp.geodesic_distance):return None
    return float(sp.geodesic_distance)

def ndtw(sim,positions,ref):
    prev=[float('inf')]*(len(ref)+1);prev[0]=0.
    for p in positions:
        cur=[float('inf')]
        for j,q in enumerate(ref):
            d=distance(sim,p,q)
            if d is None:return None
            cur.append(d+min(prev[j],prev[j+1],cur[-1]))
        prev=cur
    return math.exp(-prev[-1]/(3.*len(ref)))

def main():
    sock=socket.socket(fileno=int(sys.argv[1]));stream=sock.makefile('rw')
    split=json.loads((OUT.parent/'SPLIT.json').read_text())['closed_loop']
    jobs={j['job_id']:j for j in json.loads((DATA/'JOBS.json').read_text())}
    sim=None;scene=None;active=None;count=0;motions=0
    def send(x):stream.write(json.dumps(x)+'\n');stream.flush()
    try:
        for line in stream:
            req=json.loads(line)
            if req['op']=='close':break
            if req['op']=='reset':
                assert active is None and count<10 and req['index']==count
                row=split[count];count+=1
                active={'stage':req['stage'],'index':count-1,'job_id':row['job_id'],'scene_group':row['scene_group'],'instruction_episode_id':row['instruction_episode_id'],'started_unix':time.time()}
                append('EPISODE_LEDGER.jsonl',dict(active,event='start'))
                job=jobs[row['job_id']];e=job['episode']
                if scene!=job['scene_id']:
                    if sim:sim.close()
                    sim=simulator(job['scene_id']);scene=job['scene_id']
                sim.seed(0);s=hs.AgentState();s.position=np.asarray(e['start_position'],dtype=np.float32)
                x,y,z,w=e['start_rotation'];s.rotation=np.quaternion(w,x,y,z)
                sim.initialize_agent(0,s);obs=sim.reset()
                actual=sim.get_agent(0).get_state()
                assert np.allclose(actual.position,s.position,atol=1e-6,rtol=0)
                assert np.allclose(quaternion.as_float_array(actual.rotation),quaternion.as_float_array(s.rotation),atol=1e-6,rtol=0)
                rgb=np.ascontiguousarray(obs['rgb'][:,:,:3])
                source=json.loads((DATA/row['supervision_file']).read_text())
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==source['frames'][0]['rgb_sha256'],'INITIAL_RGB_MISMATCH'
                append('CLOSED_LOOP_INTERFACE_CHECKS.jsonl',dict(active,initial_rgb_exact=True,initial_pose_match=True,policy_response_fields=['rgb']))
                positions=[actual.position.tolist()];goal=e['goals'][0]['position'];collisions=0;motions=0
                distances=[distance(sim,positions[0],goal)]
            elif req['op']=='action':
                assert active is not None
                a=req['action'];assert a in ['move_forward','turn_left','turn_right','STOP']
                if a=='STOP' or motions>=512:
                    stopped=a=='STOP';d=distances[-1]
                    metrics=dict(active,event='complete',stopped=stopped,motion_actions=motions,logical_stops=int(stopped),action_limit=not stopped,service_failure=False,
                        stopped_goal_distance_geodesic_m=d if stopped else None,terminal_goal_distance_geodesic_m=d,
                        terminal_goal_distance_euclidean_m=float(np.linalg.norm(np.asarray(positions[-1])-goal)),
                        ever_within_3m=any(x is not None and x<3 for x in distances),stopped_within_3m=stopped and d is not None and d<3,
                        collisions=collisions,geodesic_ndtw=ndtw(sim,positions,e['reference_path']),official_sr=None,official_spl=None,
                        traveled_m=sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(positions,positions[1:])),wall_seconds=time.time()-active['started_unix'])
                    append('EPISODE_LEDGER.jsonl',metrics);active=None;send({'done':True});continue
                obs=sim.step(a);motions+=1;collisions+=int(obs['collided'])
                position=sim.get_agent(0).get_state().position.tolist();positions.append(position);distances.append(distance(sim,position,goal))
                append('SIM_STEP_LEDGER.jsonl',dict(stage=active['stage'],index=active['index'],motion=motions,action=a,position=position,collided=bool(obs['collided'])))
                rgb=np.ascontiguousarray(obs['rgb'][:,:,:3])
            else:raise ValueError('Unknown operation')
            assert rgb.shape==(224,224,3)
            send({'rgb':base64.b64encode(rgb.tobytes()).decode()})
    except BaseException as ex:
        append('SIM_FAILURES.jsonl',dict(active or {},error=repr(ex),traceback=traceback.format_exc()))
        if active:append('EPISODE_LEDGER.jsonl',dict(active,event='failure',service_failure=True,motion_actions=motions,error=repr(ex)))
        try:send({'error':repr(ex)})
        except Exception:pass
        raise
    finally:
        if sim:sim.close()
        stream.close();sock.close()

if __name__=='__main__':main()

