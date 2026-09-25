"""Re-execute moving SEE2 histories under the public backbone's camera.

The old checker sees its original audit-only 224/HFOV90 sensor. A separate
640x480/HFOV79 RGB+semantic pair verifies that its witnesses are visible to the
new policy. No old label/admission is overwritten and no Qwen model is loaded.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def main(run):
    run.mkdir(parents=True,exist_ok=False);u.setup_imports(run)
    source=u.PROJECT/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json'
    source_data=u.read(source);families=source_data['families']
    # These six previously exposed moving families are a compatibility/debug
    # pilot, not a new held-out benchmark and not the stationary scale20 pool.
    compiler_path=u.PROJECT/'data_pipeline/mechanism_factory_v2/compiler.py'
    spec=importlib.util.spec_from_file_location('strong_see2_checker',compiler_path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    protocol=dict(source=str(source),source_sha256=u.sha(source),compiler_sha256=u.sha(compiler_path),worker_sha256=u.sha(__file__),
        families=[f['family_id'] for f in families],planned_traces=sum(len(f['candidate']['histories'])*len(f['candidate']['continuations']) for f in families),
        policy_sensor=dict(height=480,width=640,hfov=79),audit_sensor=dict(height=224,width=224,hfov=90),
        policy_witness_min_pixels=math.ceil(256*640*480/(224*224)),
        old_see2_definition_unchanged=True,policy_witness_rule='Additional visibility check at both consecutive physical frames and registered 4-step model-query times; no label relaxation',
        scope='Previously exposed six moving families; debug/data compatibility only',max_decisions_per_trace=500,
        max_wall_seconds=3600,max_artifact_bytes=20*1024**3,old_training_admission_unchanged=True,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),model_loads=0,optimizer_updates=0)
    u.write(run/'PROTOCOL.json',protocol)
    import numpy as np
    import quaternion
    import habitat_sim as hs
    content=run/'content';content.mkdir();stored=set();bytes_written=0;began=time.time();complete=0;all_results=[]
    def store(array,kind):
        nonlocal bytes_written
        array=np.ascontiguousarray(array);digest=hashlib.sha256(array.tobytes()).hexdigest();key=(digest,kind)
        if key not in stored:
            if bytes_written+array.nbytes>protocol['max_artifact_bytes']:raise RuntimeError('ARTIFACT_LIMIT')
            path=content/(digest+'.'+kind+'.npy');np.save(path,array,allow_pickle=False)
            stored.add(key);bytes_written+=path.stat().st_size
        return digest
    def progress(**extra):
        u.write(run/'STATUS.json',dict(status='RUNNING',completed_traces=complete,planned_traces=protocol['planned_traces'],
            artifact_bytes=bytes_written,wall_seconds=time.time()-began,model_loads=0,optimizer_updates=0,**extra))
    progress();sim=None
    try:
        for family in families:
            cfg=hs.SimulatorConfiguration();cfg.scene_id=family['scene'];cfg.gpu_device_id=0;cfg.enable_physics=False;cfg.allow_sliding=True
            ac=hs.agent.AgentConfiguration();ac.height=1.5;ac.radius=.1;sensors=[]
            for name,kind,res,hfov in [('policy_rgb',hs.SensorType.COLOR,[480,640],79),('policy_semantic',hs.SensorType.SEMANTIC,[480,640],79),
                    ('audit_semantic',hs.SensorType.SEMANTIC,[224,224],90)]:
                sensor=hs.CameraSensorSpec();sensor.uuid=name;sensor.sensor_type=kind;sensor.resolution=res
                sensor.position=[0,1.25,0];sensor.orientation=[0,0,0];sensor.hfov=hfov;sensor.gpu2gpu_transfer=False;sensors.append(sensor)
            ac.sensor_specifications=sensors
            names={'F':'move_forward','L':'turn_left','R':'turn_right'}
            ac.action_space={name:hs.agent.ActionSpec(name,hs.agent.ActuationSpec(amount=.25 if code=='F' else 15)) for code,name in names.items()}
            sim=hs.Simulator(hs.Configuration(cfg,[ac]));compiler=mod.Compiler(**family['compiler'])
            # Source instance IDs are evaluator-only and must still exist.
            present={int(o.id.rsplit('_',1)[1]) for o in sim.semantic_scene.objects if o is not None}
            assert all(set(ids)<=present for ids in compiler.eligible.values()),'SEMANTIC_INSTANCE_MAPPING_CHANGED'
            folder=run/family['family_id'];folder.mkdir();trace_results=[];prefixes={}
            def observation(raw,step):
                rgb=np.asarray(raw['policy_rgb'][:,:,:3],dtype=np.uint8)
                semantic=np.asarray(raw['policy_semantic'],dtype=np.uint32);audit=np.asarray(raw['audit_semantic'],dtype=np.uint32)
                ids,counts=np.unique(audit,return_counts=True);pids,pcounts=np.unique(semantic,return_counts=True)
                state=sim.get_agent(0).get_state()
                return dict(step=step,evidence_complete=True,pixels={str(int(k)):int(v) for k,v in zip(ids,counts)},
                    policy_pixels={str(int(k)):int(v) for k,v in zip(pids,pcounts)},
                    rgb_hash=store(rgb,'rgb'),semantic_hash=store(semantic,'semantic'),audit_semantic_hash=store(audit,'audit_semantic'),
                    position=state.position.tolist(),rotation=quaternion.as_float_array(state.rotation).tolist())
            for history_id,history in family['candidate']['histories'].items():
                initial=family['initial_states'][history_id]
                for continuation,suffix in family['candidate']['continuations'].items():
                    actions=list(history)+list(suffix)
                    assert len(actions)<=500 and actions[-1]=='S' and 'S' not in actions[:-1]
                    sim.seed(0);state=hs.AgentState();state.position=np.asarray(initial['position'],dtype=np.float32)
                    state.rotation=np.quaternion(*initial['rotation']);sim.initialize_agent(0,state);raw=sim.reset()
                    trace=dict(actions=[],observations=[observation(raw,0)],collisions=0,complete=False,interior_state_assignments=0)
                    for action in actions:
                        if time.time()-began>protocol['max_wall_seconds']:raise RuntimeError('WALL_LIMIT')
                        trace['actions'].append(action)
                        if action!='S':
                            raw=sim.step(names[action]);trace['collisions']+=int(raw.get('collided',False))
                            trace['observations'].append(observation(raw,len(trace['observations'])))
                        if len(trace['actions'])%40==0:progress(family=family['family_id'],history=history_id,continuation=continuation,decisions=len(trace['actions']))
                    trace['complete']=True;cut=len(history);obs=trace['observations'];atoms=compiler.atoms(obs)
                    query_times=list(range(0,cut+1,4))
                    if query_times[-1]!=cut:query_times.append(cut)
                    visible={};queried={};threshold=protocol['policy_witness_min_pixels']
                    for role,eligible in compiler.eligible.items():
                        visible[role]=any(any(obs[t-1]['policy_pixels'].get(str(i),0)>=threshold and obs[t]['policy_pixels'].get(str(i),0)>=threshold for i in eligible) for t in range(1,cut+1))
                        queried[role]=any(any(obs[a]['policy_pixels'].get(str(i),0)>=threshold and obs[b]['policy_pixels'].get(str(i),0)>=threshold for i in eligible) for a,b in zip(query_times,query_times[1:]))
                    prefix=[o['rgb_hash'] for o in obs[:cut+1]]
                    if history_id in prefixes:assert prefixes[history_id]==prefix,'SAME_HISTORY_REPLAY_DIVERGED'
                    else:prefixes[history_id]=prefix
                    labels={task:compiler.evaluate(trace,task) for task in compiler.tasks}
                    labels['task_T']='unknown' if not compiler.complete(trace) else 'pass' if atoms[-1]['terminal'] else 'fail'
                    required={role:any(bool(e[role]) for e in atoms[:cut+1]) for role in compiler.eligible}
                    missing=[role for role,seen in required.items() if seen and not queried[role]]
                    path=folder/(history_id+'__'+continuation+'.json');u.write(path,trace)
                    trace_results.append(dict(history=history_id,continuation=continuation,path=str(path),sha256=u.sha(path),
                        labels=labels,collisions=trace['collisions'],decisions=len(actions),forward_actions=actions.count('F'),
                        old_event_seen=required,policy_witness_visible=visible,policy_query_witness_visible=queried,
                        missing_policy_witnesses=missing,cutoff=cut,cutoff_rgb_hash=obs[cut]['rgb_hash']))
                    complete+=1;progress(family=family['family_id'])
            shared_current=len({v[-1] for v in prefixes.values()})==1
            eligible=all(r['collisions']==0 and not r['missing_policy_witnesses'] and all(v!='unknown' for v in r['labels'].values()) for r in trace_results) and shared_current
            result=dict(family_id=family['family_id'],house=family['house'],original_partition=family['partition'],
                shared_current_rgb=shared_current,debug_candidate_eligible=eligible,training_admission=False,
                reason='Real new camera replay; only candidate audit. StreamVLN forced-history feature and action-fork checks still required.',
                traces=trace_results,source_training_admission=family['training_admission'],source_unchanged=True)
            u.write(folder/'AUDIT.json',result);all_results.append(result);sim.close();sim=None
        u.write(run/'RESULT.json',dict(status='CAMERA_REVALIDATION_COMPLETE',families=all_results,complete_traces=complete,
            eligible_debug_candidates=sum(x['debug_candidate_eligible'] for x in all_results),training_admission=False,
            method_benefit='NOT_MEASURED',wall_seconds=time.time()-began,artifact_bytes=bytes_written))
        u.write(run/'STATUS.json',dict(status='COMPLETE',completed_traces=complete,planned_traces=protocol['planned_traces'],
            eligible_debug_candidates=sum(x['debug_candidate_eligible'] for x in all_results),model_loads=0,optimizer_updates=0))
    except BaseException as error:
        u.write(run/'STATUS.json',dict(status='FAILED',error=repr(error),completed_traces=complete,planned_traces=protocol['planned_traces']))
        raise
    finally:
        if sim is not None:sim.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(u.HERE/'data_runs'/a.run_id)
