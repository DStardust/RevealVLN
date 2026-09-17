"""Reference-route compiler and independent actual replay, no learned policy."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import habitat_sim as hs
import numpy as np
import quaternion
from PIL import Image

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
ACTIONS={'move_forward':.25,'turn_left':15.,'turn_right':15.}


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(obj,f,ensure_ascii=False,allow_nan=False,indent=2)


def digest(b): return hashlib.sha256(b).hexdigest()


def position_rotation(sim):
    s=sim.get_agent(0).get_state()
    return {'position':s.position.tolist(),'rotation':quaternion.as_float_array(s.rotation).tolist()}


def record(sim,obs):
    rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]); sem=np.ascontiguousarray(obs['semantic'])
    if rgb.shape!=(224,224,3) or sem.shape!=(224,224) or rgb.dtype!=np.uint8 or float(np.std(rgb))==0:
        raise Reject('INVALID_OBSERVATION')
    state=sim.get_agent(0).get_state()
    a,b=state.sensor_states['rgb'],state.sensor_states['semantic']
    if not np.array_equal(a.position,b.position) or not np.array_equal(quaternion.as_float_array(a.rotation),quaternion.as_float_array(b.rotation)):
        raise Reject('SENSOR_MISALIGNMENT')
    return dict(position_rotation(sim),rgb_sha256=digest(rgb.tobytes()),semantic_sha256=digest(sem.tobytes())),rgb.copy()


class Reject(Exception): pass


def distance(sim,p,q):
    sp=hs.ShortestPath();sp.requested_start=np.asarray(p,dtype=np.float32);sp.requested_end=np.asarray(q,dtype=np.float32)
    if not sim.pathfinder.find_path(sp) or not math.isfinite(sp.geodesic_distance): raise Reject('UNREACHABLE_WAYPOINT')
    return float(sp.geodesic_distance),np.asarray(sp.points,dtype=np.float64)


def polyline_distance(p,points):
    if len(points)<2:return float(np.linalg.norm(np.asarray(p)-points[0]))
    a,b=points[:-1],points[1:]; v=b-a
    t=np.clip(np.sum((p-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-20),0,1)
    return float(np.linalg.norm(p-(a+t[:,None]*v),axis=1).min())


def initialize(sim,e,counts):
    sim.seed(0);s=hs.AgentState();s.position=np.asarray(e['start_position'],dtype=np.float32)
    x,y,z,w=e['start_rotation'];s.rotation=np.quaternion(w,x,y,z)
    sim.initialize_agent(0,s);obs=sim.reset();counts['resets']+=1
    st=sim.get_agent(0).get_state()
    if not np.allclose(st.position,s.position,rtol=0,atol=1e-6):raise Reject('START_POSITION_CHANGED')
    if not np.allclose(quaternion.as_float_array(st.rotation),quaternion.as_float_array(s.rotation),rtol=0,atol=1e-6):raise Reject('START_ROTATION_CHANGED')
    return obs


def simulator(scene):
    cfg=hs.SimulatorConfiguration();cfg.scene_id=str(ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{scene}/{scene}.glb')
    cfg.gpu_device_id=3;cfg.enable_physics=False;cfg.allow_sliding=False
    ac=hs.agent.AgentConfiguration();ac.height=1.5;ac.radius=.1
    ac.sensor_specifications=[]
    for name,kind in [('rgb',hs.SensorType.COLOR),('semantic',hs.SensorType.SEMANTIC)]:
        s=hs.SensorSpec();s.uuid=name;s.sensor_type=kind;s.resolution=[224,224];s.position=[0,1.25,0];s.orientation=[0,0,0]
        s.parameters['hfov']='90';s.gpu2gpu_transfer=False;ac.sensor_specifications.append(s)
    ac.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=value)) for name,value in ACTIONS.items()}
    sim=hs.Simulator(hs.Configuration(cfg,[ac]))
    if not sim.pathfinder.is_loaded: sim.close();raise RuntimeError('NAVMESH_NOT_LOADED')
    return sim


def run_job(sim,j,counts):
    e=j['episode'];actions=[];records=[];images=[];waypoints=[];max_offset=0.;collision_count=0
    out=OUT/'routes'/j['job_id'];out.mkdir(parents=True,exist_ok=False)
    started=time.time();result={'job_id':j['job_id'],'scene_id':j['scene_id'],'status':'REJECTED'}
    try:
        obs=initialize(sim,e,counts)
        r,img=record(sim,obs);records.append(r);images.append(img);counts['observations']+=1
        follower=hs.GreedyGeodesicFollower(sim.pathfinder,sim.get_agent(0),goal_radius=.35)
        targets=list(e['reference_path'])+[e['goals'][0]['position']]
        for k,target in enumerate(targets):
            target=np.asarray(target,dtype=np.float32)
            dist,line=distance(sim,records[-1]['position'],target)
            while True:
                current=np.asarray(records[-1]['position']);dist,_=distance(sim,current,target)
                euclidean=float(np.linalg.norm(current-target))
                if dist<=.35 and euclidean<=.35:
                    waypoints.append({'index':k,'decision':len(actions),'geodesic_error':dist,'euclidean_error':euclidean});break
                if len(actions)>=512:raise Reject('ACTION_BUDGET')
                before=position_rotation(sim)
                try:a=follower.next_action_along(target)
                except hs.errors.GreedyFollowerError:raise Reject('FOLLOWER_ERROR')
                if before!=position_rotation(sim):raise Reject('PLANNER_MUTATED_REAL_STATE')
                if a not in ACTIONS:raise Reject('EARLY_FOLLOWER_STOP')
                actions.append(a);obs=sim.step(a);counts['primitive_actions']+=1
                r,img=record(sim,obs);records.append(r);images.append(img);counts['observations']+=1
                if obs['collided']:
                    collision_count+=1;counts['collisions']+=1;raise Reject('COLLISION')
                offset=polyline_distance(np.asarray(r['position']),line);max_offset=max(max_offset,offset)
                if offset>.75:raise Reject('REFERENCE_CORRIDOR_DEVIATION')
        actions.append('STOP');counts['logical_stops']+=1
        # No teleport within a trajectory. New initialization starts a second full replay.
        obs=initialize(sim,e,counts)
        replay_records=[]
        for t,a in enumerate(actions):
            rr,_=record(sim,obs);replay_records.append(rr);counts['observations']+=1
            if rr!=records[t]:raise Reject('REPLAY_PIXEL_OR_POSE_MISMATCH')
            if a=='STOP':counts['logical_stops']+=1;break
            obs=sim.step(a);counts['primitive_actions']+=1
            if obs['collided']:counts['collisions']+=1;raise Reject('REPLAY_COLLISION')
        save(out/'replay_certificate.json',{'exact_all_frame_pose_rgb_semantic_match':True,
             'frames_compared':len(replay_records),'seed':0,'replay_count':2})
        rgb_refs=[];content=OUT/'content';content.mkdir(exist_ok=True)
        for r,img in zip(records,images):
            p=content/f'{r["rgb_sha256"]}.png'
            if not p.exists():
                tmp=p.with_suffix('.png.tmp');Image.fromarray(img).save(tmp,format='PNG');tmp.replace(p)
            rgb_refs.append(str(p.relative_to(OUT)))
        supervision={'actions':actions,'frames':records,'waypoint_checks':waypoints,
                      'source_episode_id':e['episode_id'],'scene_id':j['scene_id'],
                      'physical_source_route_sha256':j['physical_source_route_sha256'],
                      'label_provenance':'reference_waypoint_follower_actual_replay',
                      'natural_language_full_semantics_certified':False}
        save(out/'supervision_only.json',supervision)
        for alias in j['instruction_alias_episodes']:
            save(out/f'policy_{alias["episode_id"]}.json',{'instruction':alias['instruction']['instruction_text'],'rgb_sequence':rgb_refs})
        result.update(status='CERTIFIED',instruction_records=len(j['instruction_alias_episodes']),
                      decisions=len(actions),frame_count=len(records),replay_pass=True)
    except Reject as ex:
        result['reason']=str(ex)
    finally:
        save(out/'diagnostic.json',{'actions_attempted':actions,'observations':records,'waypoints':waypoints,
             'collision_count':collision_count,'maximum_segment_corridor_offset_m':max_offset})
    result.update(elapsed_seconds=time.time()-started,recorded_observations=len(records))
    save(out/'result.json',result)
    with (OUT/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    jobs=json.loads((OUT/'JOBS.json').read_text());assert len(jobs)==100
    assert not (OUT/'LEDGER.jsonl').exists(),'Closed or interrupted node must not be overwritten'
    counts={'primitive_actions':0,'observations':0,'resets':0,'logical_stops':0,'collisions':0,'simulator_constructions':0}
    results=[];sim=None;scene=None;started=time.time()
    try:
        for j in jobs:
            if scene!=j['scene_id']:
                if sim:sim.close();sim=None
                sim=simulator(j['scene_id']);scene=j['scene_id'];counts['simulator_constructions']+=1
            results.append(run_job(sim,j,counts))
            # Persist counts even if a later job raises an unexpected implementation error.
            (OUT/'COUNTS_LIVE.json').write_text(json.dumps(counts,indent=2))
    finally:
        if sim:sim.close()
        save(OUT/'ACTUAL_COUNTS.json',dict(counts,wall_seconds=time.time()-started,completed_candidates=len(results)))
    passed=[r for r in results if r['status']=='CERTIFIED']
    per_scene={s:sum(r['status']=='CERTIFIED' for r in results if r['scene_id']==s) for s in sorted({j['scene_id'] for j in jobs})}
    save(OUT/'GENERATION_RESULT.json',{'attempted':len(results),'certified_routes':len(passed),
         'certified_per_scene':per_scene,'instruction_records':sum(r['instruction_records'] for r in passed),
         'unique_route_decisions':sum(r['decisions'] for r in passed),
         'generation_yield_pass':len(passed)>=80 and min(per_scene.values())>=10,
         'integrity_acceptance':'PENDING_AUDIT','scientific_pass':False,'navigation_gain':None})


if __name__=='__main__': main()
