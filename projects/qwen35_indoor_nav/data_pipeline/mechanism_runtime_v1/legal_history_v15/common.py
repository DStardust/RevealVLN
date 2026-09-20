"""Shared paths and raw trace measurements; no simulator imported here."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
RESEARCH = LINE/'research/continuation_memory_v1/legal_closed_loop_v15'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = load('v15_io', LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/common.py')


def raw_window(trace, cutoff):
    return dict(rgb=[o['rgb_hash'] for o in trace['observations'][max(0, cutoff-1):cutoff+1]],
                executed=trace['actions'][max(0, cutoff-8):cutoff])


def pose_delta(left, right):
    values = [abs(a-b) for key in ('position', 'rotation') for a, b in zip(left[key], right[key])]
    for sensor in left['sensors']:
        values.append(pose_delta(dict(left['sensors'][sensor], sensors={}),
                                 dict(right['sensors'][sensor], sensors={})))
    return max(values)


def exact_pose_equal(left,right):
    """Exact physical pose: q and -q encode one rotation; no tolerance/snapping."""
    return (left['position']==right['position'] and
        (left['rotation']==right['rotation'] or left['rotation']==[-x for x in right['rotation']]) and
        left.get('sensors',{}).keys()==right.get('sensors',{}).keys() and
        all(exact_pose_equal(left['sensors'][k],right['sensors'][k]) for k in left.get('sensors',{})))
