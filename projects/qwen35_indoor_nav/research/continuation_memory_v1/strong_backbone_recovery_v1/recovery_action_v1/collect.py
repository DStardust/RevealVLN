"""Replay actual histories, then physically execute an offline goal teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import common as u


def main(run, output, ids):
    if (run/'SOURCE_LOCK.json').exists():u.verify_sources(run)
    output.mkdir(parents=True,exist_ok=False);u.setup_imports(output)
    import numpy as np
    import habitat
    from habitat_extensions import measures  # official OracleSuccess registration
    from habitat_baselines.config.default import get_config
    from habitat.config import read_write
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    def image_hash(x):
        x=np.ascontiguousarray(x);h=hashlib.sha256(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes());return h.hexdigest()
    planned={e['id']:e for e in u.read(run/'DATA_MANIFEST.json')['episodes']};completed=[]
    for index in ids:
        e=planned[index];assert u.sha(e['source_trace'])==e['source_trace_sha256']
        assert u.sha(e['config'])==e['config_sha256']
        old=[json.loads(x) for x in Path(e['source_trace']).read_text().splitlines()]
        original=[x for x in old if x['event']=='action'];reset=next(x for x in old if x['event']=='reset')
        cfg=get_config(e['config'])
        with read_write(cfg):cfg.habitat.task.measurements.update({'collisions':CollisionsMeasurementConfig()})
        folder=output/'episodes'/str(index);folder.mkdir(parents=True)
        env=habitat.Env(config=cfg);began=time.time();actions=[];observations=[];events=[]
        try:
            obs=env.reset();assert str(env.current_episode.episode_id)==str(e['episode_id'])
            assert env.current_episode.instruction.instruction_text==e['instruction']
            assert image_hash(obs['rgb'])==reset['rgb_sha256'],'INITIAL_RGB_MISMATCH'
            teacher=ShortestPathFollower(env.sim,goal_radius=1.8,return_one_hot=False)
            prefix=len(original) if e['kind']=='PRESERVATION' else e['cutoff']
            failure=None
            while not env.episode_over:
                step=len(actions);rgb=image_hash(obs['rgb'])
                if step<prefix:
                    assert rgb==original[step]['before_rgb_sha256'],'PREFIX_RGB_MISMATCH'
                    action=original[step]['executed_action']
                else:
                    try:action=teacher.get_next_action(env.current_episode.goals[0].position)
                    except Exception as error:
                        failure=type(error).__name__+':'+str(error);break
                    if action is None:failure='FOLLOWER_NO_ACTION';break
                assert int(action) in range(4)
                observations.append(rgb);actions.append(int(action));obs=env.step(int(action))
                metrics=env.get_metrics()
                events.append(dict(step=len(actions),action=int(action),distance=float(metrics['distance_to_goal']),
                    collision=bool(metrics['collisions']['is_collision']),teacher=step>=prefix))
                if step<prefix:assert image_hash(obs['rgb'])==original[step]['after_rgb_sha256'],'PREFIX_TRANSITION_MISMATCH'
            metrics=env.get_metrics();success=bool(metrics['success']) and bool(actions) and actions[-1]==0
            assert len(actions)<=500
            if e['kind']=='PRESERVATION':assert success and len(actions)==len(original),'PRESERVATION_REPLAY_CHANGED'
            # Actual native chunk boundaries before takeover; teacher chunks end
            # at four actions, frame reset, or immediately before terminal STOP.
            queries=[x['environment_step'] for x in old if x['event']=='generation' and x['environment_step']<prefix]
            if e['kind']=='PRESERVATION':queries=[x['environment_step'] for x in old if x['event']=='generation']
            else:
                q=prefix
                while q<len(actions):
                    queries.append(q);end=min(q+4,((q//32)+1)*32,len(actions))
                    if q<len(actions)-1<end and actions[-1]==0:end=len(actions)-1
                    q=end
            assert len(queries)==len(set(queries)) and queries==sorted(queries)
            record=dict(id=index,source_id=e['source_id'],house=e['house'],partition=e['partition'],kind=e['kind'],
                cutoff=prefix,actions=actions,rgb_sha256=observations,query_steps=queries,events=events,
                success=success,admitted=success and failure is None,teacher_error=failure,
                native_success=bool(e['native_outcome']['success']),full_budget=500,
                prefix_rgb_exact=True,teleport=False,teacher_oracle_used_offline=e['kind']=='RECOVERY',
                terminal_distance=float(metrics['distance_to_goal']),wall_seconds=time.time()-began)
            u.write(folder/'TRAJECTORY.json',record)
            u.write(folder/'COMPLETE.json',dict(id=index,admitted=record['admitted'],kind=e['kind'],
                trajectory=str(folder/'TRAJECTORY.json'),sha256=u.sha(folder/'TRAJECTORY.json')))
        finally:env.close()
        completed.append(index);u.write(output/'PROGRESS.json',dict(status='RUNNING',complete=completed,planned=ids))
    u.write(output/'STATE_SEAL.json',dict(no_model_loaded=True,all_histories_actually_executed=True))
    u.write(output/'RESULT.json',dict(status='COMPLETE',complete=completed,planned=ids))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',required=True)
    a=p.parse_args();main(a.run,a.output,[int(x) for x in a.ids.split(',')])
