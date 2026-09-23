"""Bounded renderer-only diagnosis at registered initial states, no agent actions."""
import hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import common as u
import habitat_sim as hs
import numpy as np
import quaternion
from PIL import Image
OUT=HERE/'rgb_turn_probe_001';OUT.mkdir(exist_ok=False)
sys.argv=[sys.argv[0],'-1',str(OUT),'PROBE']
old=u.load('rgb_probe_frozen_service',u.ASSET/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/executor.py')
p=u.read(u.HERE/'PROTOCOL.json');p['gpu']=1;episodes=u.read(u.HERE/'manifests/FIT_EPISODES.json');results=[];start=time.monotonic()
def collect(sim,index,tag):
    e=episodes[index];sim.seed(0);state=hs.AgentState();state.position=np.asarray(e['start_position'],dtype=np.float32);state.rotation=np.quaternion(*u.c.xyzw_to_wxyz(e['start_rotation']));sim.initialize_agent(0,state);obs=sim.reset();initial=sim.get_agent(0).get_state()
    for j in range(3):
        if j:obs=sim.get_sensor_observations()
        rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]);Image.fromarray(rgb).save(OUT/f'{tag}_{index}_{j}.png')
        sensor=sim.get_agent(0)._sensors['rgb']
        results.append(dict(tag=tag,index=index,episode_id=e['episode_id'],read=j,shape=list(rgb.shape),dtype=str(rgb.dtype),min=int(rgb.min()),max=int(rgb.max()),std=float(rgb.std()),sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),position=initial.position.tolist(),rotation=quaternion.as_float_array(initial.rotation).tolist(),camera_transform=np.asarray(sensor.node.absolute_transformation()).tolist(),agent_navigable=sim.pathfinder.is_navigable(initial.position),camera_height=p['camera_height']))
house=episodes[139]['scene_id'].split('/')[-2]
sim=old.build_sim(house,p)
try:
    collect(sim,139,'fresh')
    for step in range(1,7):
        obs=sim.step('turn_left');rgb=np.ascontiguousarray(obs['rgb'][:,:,:3]);Image.fromarray(rgb).save(OUT/f'turn_{step}.png');results.append(dict(diagnostic_action='turn_left',step=step,std=float(rgb.std()),min=int(rgb.min()),max=int(rgb.max()),position=sim.get_agent(0).get_state().position.tolist(),sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
finally:sim.close()
sim=old.build_sim(house,p)
try:
    collect(sim,136,'control');collect(sim,139,'warm')
finally:sim.close()
u.write(OUT/'RESULT.json',dict(rows=results,wall_seconds=time.monotonic()-start,model_forwards=0,navigation_actions=6,scene=house,gpu=1,semantics='diagnostic-only six physical left turns from failed initial state; not a training or evaluation trajectory'),True)
print(json.dumps(results),flush=True)
