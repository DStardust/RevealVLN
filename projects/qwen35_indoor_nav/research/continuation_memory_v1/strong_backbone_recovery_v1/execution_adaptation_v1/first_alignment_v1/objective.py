"""Train only the action positions that the FIRST policy can change."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ADAPT = HERE.parent
sys.path[:0] = [str(ADAPT), str(ADAPT.parent)]
spec = importlib.util.spec_from_file_location('frozen_all_token_trainer', ADAPT / 'train.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
all_position_objective = legacy.objective


def first_positions(row):
    """Keep every physical memory update; select only query-start actor rows."""
    mask = row['action_token_offsets'] == 0
    if not bool(mask.any()):
        raise ValueError('EMPTY_FIRST_POSITION')
    result = dict(row)
    for key in ('actor_features', 'base_logits', 'actor_context_steps',
                'actor_environment_steps', 'action_token_offsets', 'known',
                'targets', 'supervision_region'):
        result[key] = row[key][mask]
    # executed_actions and memory_features still cover the complete real history.
    return result


def objective(model, recovery, ordinary, weights, device):
    r, o = first_positions(recovery), first_positions(ordinary)
    loss, info = all_position_objective(model, r, o, weights, device)
    info.update(supervision_scope='FIRST_ONLY',
        memory_observations=len(recovery['memory_features']) + len(ordinary['memory_features']),
        excluded_actor_positions=len(recovery['targets']) + len(ordinary['targets']) - len(r['targets']) - len(o['targets']))
    return loss, info
