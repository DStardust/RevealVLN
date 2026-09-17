"""Deterministic finite-vocabulary surface wording; structural programs untouched."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WF = HERE.parent
RUNTIME = WF.parent
ROOT = RUNTIME.parents[3]
PLANNER = RUNTIME.parent / 'mechanism_scale_v1/planner.py'
spec = importlib.util.spec_from_file_location('language_registered_vocabulary', PLANNER)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
VERSION = 'q35n.finite_language_realization.v1'
# These two expansions are already explicit in feedback_generation_v1/prepare.py.
ROOM_EXPANSIONS = {'tv': 'TV room', 'familyroom/lounge': 'family room/lounge'}
# Do not silently infer that a room labelled "toilet" is a bathroom.
EXPLICIT_ROOM_TYPES = {'toilet'}
IDENTITY_ROOMS = planner.INDOOR_ROOMS - set(ROOM_EXPANSIONS) - EXPLICIT_ROOM_TYPES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def room_phrase(room, unknown='reject'):
    if unknown not in ('reject', 'explicit_type'):
        raise ValueError('UNKNOWN_ROOM_POLICY')
    if not isinstance(room, str) or not room or any(c in room for c in '\n\r\t()<>'):
        raise ValueError('INVALID_ROOM_LABEL')
    if room in ROOM_EXPANSIONS:
        return ROOM_EXPANSIONS[room]
    if room in IDENTITY_ROOMS:
        return room
    if room in EXPLICIT_ROOM_TYPES or unknown == 'explicit_type':
        return 'room (room type: ' + room + ')'
    raise ValueError('UNREGISTERED_ROOM_LABEL')


def role_phrase(role, unknown_room='reject'):
    if not isinstance(role, dict) or set(role) != {'mpcat40', 'room', 'raw_match'}:
        raise ValueError('ROLE_SCHEMA')
    raw_match = role['raw_match']
    if not isinstance(raw_match, dict) or set(raw_match) != {'mode', 'value'} or raw_match['mode'] != 'exact':
        raise ValueError('EXACT_RAW_MATCH_REQUIRED')
    raw = raw_match['value']
    if raw not in planner.RAW.get(role['mpcat40'], set()):
        raise ValueError('UNREGISTERED_CATEGORY_RAW_PAIR')
    # Lexical capitalization only; never replace a raw subtype by its broad category.
    noun = 'TV' if raw == 'tv' else raw
    return 'the ' + noun + ' in the ' + room_phrase(role['room'], unknown_room)


def structure(tasks):
    return {name: {k: v for k, v in task.items() if k != 'instruction'} for name, task in tasks.items()}


def realize_tasks(roles, tasks, unknown_room='reject'):
    """Return a fresh task mapping. Only instruction strings may change."""
    if not isinstance(roles, dict) or not roles or not isinstance(tasks, dict) or not tasks:
        raise ValueError('ROLES_TASKS_REQUIRED')
    for role in roles.values():
        role_phrase(role, unknown_room)
    result = copy.deepcopy(tasks)
    for name, task in result.items():
        if not isinstance(name, str) or not name or not isinstance(task, dict) or set(task) != {'anchor', 'terminal', 'instruction'}:
            raise ValueError('TASK_SCHEMA')
        if task['anchor'] not in roles or task['terminal'] not in roles or not isinstance(task['instruction'], str) or not task['instruction']:
            raise ValueError('TASK_ROLE_REFERENCE')
        task['instruction'] = ('First see ' + role_phrase(roles[task['anchor']], unknown_room)
            + ' in two consecutive observations, then see '
            + role_phrase(roles[task['terminal']], unknown_room)
            + ' in two consecutive observations, and stop immediately.')
    assert structure(result) == structure(tasks)
    return result


def propose_revision(candidate):
    tasks = realize_tasks(candidate['roles'], candidate['tasks'])
    changes = [dict(task_id=k, old_instruction=v['instruction'], new_instruction=tasks[k]['instruction'])
        for k, v in candidate['tasks'].items() if v['instruction'] != tasks[k]['instruction']]
    return dict(schema_version=VERSION, candidate_id=candidate['candidate_id'], house_id=candidate['house_id'],
        proposed_tasks=tasks, changes=changes,
        source_tasks_sha256=digest(candidate['tasks']), proposed_tasks_sha256=digest(tasks),
        invariant_role_structure_sha256=digest(dict(roles=candidate['roles'], task_structure=structure(tasks))),
        task_program_unchanged=True, eligible_ids_unchanged=True, actions_unchanged=True,
        y_checker_unchanged=True, physical_replay_required=True, policy_instruction_hashes_must_be_regenerated=True,
        executable=False, training_admission=False)
