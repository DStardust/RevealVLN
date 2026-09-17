"""One reset and one turn; no family construction or navigation evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else CODE
LINE = CODE.parents[1]
assert OUT.is_relative_to(CODE)
PROJECT = LINE.parents[1]
OLD = CODE.parent / 'Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')

def scene_check():
    manifest = json.loads((OLD / 'SCENE_ASSET_FINGERPRINT.json').read_text())
    root = Path(manifest['root'])
    assert root.resolve().is_relative_to(PROJECT)
    for record in manifest['files']:
        p = root / record['name']
        assert p.resolve().is_relative_to(PROJECT)
        assert p.stat().st_size == record['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']
    return root / '17DRP5sb8fy.glb'

def main():
    import habitat_sim
    import magnum
    import numpy as np
    import quaternion
    from PIL import Image
    for module in (habitat_sim, magnum, np, quaternion):
        assert Path(module.__file__).resolve().is_relative_to(LINE), module.__file__
    scene = scene_check()
    gpu_id = int(sys.argv[1])
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene)
    cfg.gpu_device_id = gpu_id
    cfg.enable_physics = False
    cfg.allow_sliding = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius = 1.5, 0.1
    specs = []
    for name, sensor_type in [('rgb', habitat_sim.SensorType.COLOR), ('semantic', habitat_sim.SensorType.SEMANTIC)]:
        spec = habitat_sim.SensorSpec()
        spec.uuid, spec.sensor_type = name, sensor_type
        spec.resolution, spec.position, spec.orientation = [224,224], [0,1.25,0], [0,0,0]
        spec.parameters['hfov'] = '90'
        assert spec.parameters['hfov'] == '90'
        spec.gpu2gpu_transfer = False
        specs.append(spec)
    agent.sensor_specifications = specs
    agent.action_space = {name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
                          for name, amount in [('move_forward',0.25),('turn_left',15.0),('turn_right',15.0)]}
    save('SMOKE_EXECUTION_CONFIG.json', {
        'scene':str(scene), 'physical_gpu_ordinal':gpu_id, 'hfov':90,
        'resolution':[224,224], 'sensor_position':[0,1.25,0], 'sensor_orientation':[0,0,0],
        'agent_height':1.5, 'agent_radius':0.1, 'allow_sliding':False,
        'forward_m':0.25,'turn_deg':15,'seed':0,'one_action':'turn_left',
        'episode_is_navigation_evaluation':False, 'model_loads':0,
        'habitat_version':habitat_sim.__version__, 'habitat_file':habitat_sim.__file__})
    counts = {'simulator_constructions_attempted':1, 'simulator_constructions_succeeded':0,
              'explicit_resets':0, 'explicit_actions':0}
    save('SMOKE_COUNTS.json', counts)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent])) as sim:
        counts['simulator_constructions_succeeded'] = 1
        save('SMOKE_COUNTS.json', counts)
        sim.seed(0)
        assert sim.pathfinder.is_loaded
        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.get_random_navigable_point()
        state.rotation = np.quaternion(1,0,0,0)
        assert sim.pathfinder.is_navigable(state.position)
        sim.initialize_agent(0,state)
        obs0 = sim.reset()
        counts['explicit_resets'] += 1
        save('SMOKE_COUNTS.json',counts)
        obs1 = sim.step('turn_left')
        counts['explicit_actions'] += 1
        save('SMOKE_COUNTS.json',counts)
        assert isinstance(obs1['collided'], (bool,np.bool_))
        observations = []
        for t, obs in enumerate((obs0,obs1)):
            rgb, semantic = obs['rgb'], obs['semantic']
            assert rgb.shape[:2] == semantic.shape == (224,224)
            assert rgb.dtype == np.uint8 and np.issubdtype(semantic.dtype,np.integer)
            assert np.std(rgb[:,:,:3]) > 0
            assert len(np.unique(semantic)) > 1, 'Semantic output lacks variation'
            np.savez_compressed(OUT / f'observation_{t}.npz',rgb=rgb,semantic=semantic)
            Image.fromarray(rgb[:,:,:3]).save(OUT / f'rgb_{t}.png')
            observations.append({'index':t,'rgb_shape':list(rgb.shape),'semantic_shape':list(semantic.shape),
                'rgb_dtype':str(rgb.dtype),'semantic_dtype':str(semantic.dtype),
                'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                'semantic_sha256':hashlib.sha256(semantic.tobytes()).hexdigest(),
                'semantic_unique_count':int(len(np.unique(semantic)))})
        after = sim.get_agent(0).get_state()
        rgb_state, sem_state = after.sensor_states['rgb'], after.sensor_states['semantic']
        assert np.allclose(rgb_state.position, sem_state.position, atol=1e-7)
        assert np.allclose(quaternion.as_float_array(rgb_state.rotation),
                           quaternion.as_float_array(sem_state.rotation), atol=1e-7)
        assert np.allclose(state.position, after.position, atol=1e-5)
        save('SMOKE_OBSERVATIONS.json',{'observations':observations,'collided':bool(obs1['collided']),
             'start_position':state.position.tolist(),'end_position':after.position.tolist(),
             'start_rotation':quaternion.as_float_array(state.rotation).tolist(),
             'end_rotation':quaternion.as_float_array(after.rotation).tolist(),
             'sensor_extrinsics_match':True,
             'semantic_objects':len(sim.semantic_scene.objects),
             'semantic_regions':len(sim.semantic_scene.regions),
             'renderer_interface_pass':True,'semantic_task_label_acceptance':False})
        print('G0R renderer interface smoke passed; no family or science claim.',flush=True)
    scene_check()

if __name__ == '__main__':
    main()
