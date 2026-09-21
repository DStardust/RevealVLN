"""Bounded actual-rotation join probe; no state snapping or model access."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continuation_service import NoInteriorJoin, exact_pose
from collect import ContentStore
from v16_common import read, write

def main():
    root = Path(__file__).resolve().parent
    out = root / 'probe_001'
    out.mkdir(exist_ok=False)
    data = read(root.parent / 'runs/v16_fitdev_train_001/DATA.json')
    family = next(f for f in data['raw_families'] if f['split'] == 'FIT')
    backend = NoInteriorJoin(family['scene'], 1, family['roles'], ContentStore(out/'content', 2**30),
                             dict(runtime_allowed=True, scene_glb=family['scene'], gpu_device=1))
    began = time.monotonic()
    rows = []
    try:
        for direction, amount in [('L', 3), ('R', 3), ('L', 6), ('R', 6), ('L', 9), ('R', 9)]:
            backend.reset(family['initial_position'], 0, 0)
            actions = [direction]*amount + [('R' if direction=='L' else 'L')]*amount + ['L','R']*32
            trace = dict(actions=[], observations=[dict(backend.observe(), step=0)], collisions=0,
                         complete=False, interior_state_assignments=0)
            for action in actions:
                trace['collisions'] += int(backend.step(action)); trace['actions'].append(action)
                trace['observations'].append(dict(backend.observe(), step=len(trace['actions'])))
            trace['complete'] = True
            name = direction+str(amount)
            write(out/(name+'.json'), trace, True)
            last = trace['observations'][-2:]
            rows.append(dict(name=name, pose=last[-1]['pose'], rgb=[o['rgb_hash'] for o in last]))
        comparisons = [dict(a=a['name'],b=b['name'],raw_equal=a['rgb']==b['rgb'],
                            pose_equal=exact_pose(a['pose'],b['pose']))
                       for i,a in enumerate(rows) for b in rows[i+1:]]
        result = dict(family=family['family_id'], seconds=time.monotonic()-began, comparisons=comparisons,
                      backend_counts=backend.counts, new_certified_families=0)
        write(out/'RESULT.json', result, True);print(json.dumps(result),flush=True)
    finally:
        backend.close()

if __name__ == '__main__': main()
