"""Real equal-length/count histories; discover exact raw/pose joins without snapping."""
import collections
import itertools
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continuation_service import NoInteriorJoin
from collect import ContentStore
from v16_common import read, write, digest

def words():
    for left in itertools.combinations(range(12), 6):
        selected = set(left)
        yield ['L' if t in selected else 'R' for t in range(12)] + ['L','R']*4

def main():
    root = Path(__file__).resolve().parent
    out = root / 'balanced_probe_001'; out.mkdir(exist_ok=False)
    data = read(root.parent / 'runs/v16_fitdev_train_001/DATA.json')
    family = next(f for f in data['raw_families'] if f['split'] == 'FIT')
    backend = NoInteriorJoin(family['scene'], 1, family['roles'], ContentStore(out/'content', 2**30),
                             dict(runtime_allowed=True, scene_glb=family['scene'], gpu_device=1))
    began = time.monotonic(); groups = collections.defaultdict(list)
    try:
        for index, actions in enumerate(words()):
            if time.monotonic()-began > 600:raise TimeoutError('BOUNDED_PROBE_600_SECONDS')
            backend.reset(family['initial_position'], 0, 0)
            trace = dict(actions=[], observations=[dict(backend.observe(), step=0)], collisions=0,
                         complete=False, interior_state_assignments=0)
            for action in actions:
                trace['collisions'] += int(backend.step(action));trace['actions'].append(action)
                trace['observations'].append(dict(backend.observe(), step=len(trace['actions'])))
            trace['complete']=True
            write(out/f'H_{index:04d}.json',trace,True)
            last=trace['observations'][-2:]
            key=digest(dict(rgb=[o['rgb_hash'] for o in last],pose=last[-1]['pose'],actions=actions[-8:]))
            groups[key].append(index)
            if index%100==0:
                write(out/'STATUS.json',dict(completed=index+1,groups=len(groups),max_group=max(map(len,groups.values())),seconds=time.monotonic()-began))
        result=dict(completed=index+1,groups=len(groups),sizes=dict(collections.Counter(map(len,groups.values()))),
                    matched_groups={k:v for k,v in groups.items() if len(v)>1},seconds=time.monotonic()-began,
                    backend_counts=backend.counts,new_certified_families=0)
        write(out/'RESULT.json',result,True);print(json.dumps({k:v for k,v in result.items() if k!='matched_groups'}),flush=True)
    finally:backend.close()

if __name__=='__main__':main()
