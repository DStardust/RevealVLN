"""CPU-only, instance-scoped compiler. No simulator or learned encoder.

All pixel counts, entity IDs and programs stay on the supervision side.  The
policy projection is constructed explicitly; arbitrary trace keys never pass.
"""
import copy
import hashlib
import json
from types import MappingProxyType


ACTIONS = MappingProxyType({'F': 'MOVE_FORWARD', 'L': 'TURN_LEFT',
                            'R': 'TURN_RIGHT', 'S': 'STOP'})
QUERY_SCHEMA = 'q35n.continuation_query.v4'


def canonical(value):
    def keys(x):
        if isinstance(x, dict) or isinstance(x, MappingProxyType):
            if len({str(k) for k in x}) != len(x):
                raise ValueError('JSON_KEY_COLLISION')
            return {str(k): keys(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [keys(v) for v in x]
        return x
    return json.dumps(keys(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def ref(value):
    return 'sha256:' + digest(value)


def pixel(observation, instance):
    return observation['pixels'].get(str(instance), observation['pixels'].get(instance, 0))


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _known_observation(observation, step):
    if not isinstance(observation, dict):
        return False
    pixels = observation.get('pixels')
    if (type(observation.get('step')) is not int or observation['step'] != step
            or observation.get('evidence_complete') is not True or not isinstance(pixels, dict)):
        return False
    ids = set()
    for key, count in pixels.items():
        if isinstance(key, bool) or not str(key).isdigit() or str(int(key)) != str(key):
            return False
        if str(key) in ids or not _integer(count):
            return False
        ids.add(str(key))
    return True


def complete(trace):
    """Malformed/incomplete evidence is unknown, never an implicit negative."""
    if not isinstance(trace, dict):
        return False
    actions, observations = trace.get('actions'), trace.get('observations')
    if (trace.get('complete') is not True or type(trace.get('collisions')) is not int
            or trace['collisions'] != 0 or not isinstance(actions, list)
            or not isinstance(observations, list) or not observations):
        return False
    if any(not isinstance(a, str) or a not in ACTIONS for a in actions) or 'S' in actions[:-1]:
        return False
    expected = len(actions) + (0 if actions[-1:] == ['S'] else 1)
    return len(observations) == expected and all(
        _known_observation(o, t) for t, o in enumerate(observations))


def slice_continuation(trace, cutoff):
    if type(cutoff) is not int or not 0 <= cutoff < len(trace['observations']):
        raise ValueError('INVALID_CUTOFF')
    return {'actions': list(trace['actions'][cutoff:]),
            'observations': [dict(copy.deepcopy(o), step=t)
                             for t, o in enumerate(trace['observations'][cutoff:])],
            'collisions': trace['collisions'], 'complete': trace['complete']}


def semantic_query(query):
    return {k: copy.deepcopy(query[k])
            for k in ('query_schema_version', 'coordinate_frame', 'sequence')}


def policy_semantics(record):
    allowed = ('task_revision', 'task_type', 'instruction', 'causal_cutoff_step',
               'observations', 'executed_actions', 'memory_reset', 'sensor_profile')
    return {key: copy.deepcopy(record[key]) for key in allowed}


class Compiler:
    def __init__(self, roles, tasks, eligible, *,
                 task_revision='observable_see2_then_stop.v4', sensor=None):
        if not roles or set(roles) != set(eligible):
            raise ValueError('ROLE_ELIGIBILITY_KEYS')
        role_copy, eligible_copy, task_copy = {}, {}, {}
        for role, pair in roles.items():
            if (not isinstance(role, str) or not role or not isinstance(pair, (list, tuple))
                    or len(pair) != 2 or any(not isinstance(s, str) or not s for s in pair)):
                raise ValueError('ROLE_DEFINITION')
            ids = list(eligible[role])
            if any(not _integer(i) for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('INSTANCE_IDS')
            role_copy[role] = tuple(pair)
            eligible_copy[role] = tuple(sorted(ids))
        if not tasks:
            raise ValueError('NO_TASKS')
        for task_id, task in tasks.items():
            if (not isinstance(task_id, str) or not task_id or not isinstance(task, dict)
                    or set(task) != {'anchor', 'terminal', 'instruction'}
                    or task['anchor'] not in role_copy or task['terminal'] not in role_copy
                    or not isinstance(task['instruction'], str) or not task['instruction']):
                raise ValueError('TASK_DEFINITION')
            task_copy[task_id] = MappingProxyType(dict(task))
        if not isinstance(task_revision, str) or not task_revision:
            raise ValueError('TASK_REVISION')
        expected = {'rgb_height': 224, 'rgb_width': 224, 'rgb_hfov_deg': 90,
                    'sensor_position_m': [0, 1.25, 0], 'sensor_orientation_rad': [0, 0, 0],
                    'recent_rgb_frames': 2, 'recent_action_window': 8}
        if sensor is not None and canonical(sensor) != canonical(expected):
            raise ValueError('UNREGISTERED_SENSOR_REVISION')
        self.roles = MappingProxyType(role_copy)
        self.tasks = MappingProxyType(task_copy)
        self.eligible = MappingProxyType(eligible_copy)
        self.task_revision = task_revision
        self._sensor = canonical(expected)
        self.query_vocabulary = MappingProxyType({
            'categories': tuple(sorted({pair[0] for pair in role_copy.values()})),
            'rooms': tuple(sorted({pair[1] for pair in role_copy.values()}))})

    complete = staticmethod(complete)
    slice_continuation = staticmethod(slice_continuation)
    semantic_query = staticmethod(semantic_query)
    policy_semantics = staticmethod(policy_semantics)

    def atoms(self, observations):
        result = []
        for t, observation in enumerate(observations):
            if t == 0:
                result.append({role: [] for role in self.roles})
            elif not _known_observation(observation, t) or not _known_observation(observations[t-1], t-1):
                result.append({role: None for role in self.roles})
            else:
                result.append({role: [idx for idx in ids if pixel(observation, idx) >= 256
                                     and pixel(observations[t-1], idx) >= 256]
                               for role, ids in self.eligible.items()})
        return result

    def evaluate(self, trace, task_id):
        task = self.tasks[task_id]
        if not complete(trace):
            return 'unknown'
        events = self.atoms(trace['observations'])
        return 'pass' if (trace['actions'][-1:] == ['S'] and events[-1][task['terminal']]
                          and any(e[task['anchor']] for e in events[:-1])) else 'fail'

    def m2(self, trace, task_id):
        """Causal strongest program state; malformed future cannot taint past."""
        task = self.tasks[task_id]
        events, actions = self.atoms(trace['observations']), trace['actions']
        seen, known, result = False, True, []
        for t, event in enumerate(events):
            known = known and _known_observation(trace['observations'][t], t)
            if t:
                known = known and t <= len(actions) and actions[t-1] in ('F', 'L', 'R')
            ready = seen and bool(event[task['terminal']])
            seen = seen or bool(event[task['anchor']])
            state = ('UNKNOWN' if not known else 'READY_TO_STOP' if ready else
                     'WAIT_TERMINAL_WITNESS' if seen else 'WAIT_ANCHOR')
            result.append({'decision_step': t, 'state': state, 'truth_known': bool(known),
                           'loss_mask': int(known), 'causal_event_cutoff_step': t})
        return result

    def query_from_trace(self, trace):
        if not complete(trace):
            raise ValueError('INCOMPLETE_QUERY_TRACE')
        events, sequence = self.atoms(trace['observations']), []
        for t, action in enumerate(trace['actions']):
            if t:
                # Sort semantic pairs, not arbitrary role names or instance IDs.
                visible = sorted({self.roles[r] for r in self.roles if events[t][r]})
                for category, room in visible:
                    sequence.append({'kind': 'observe', 'object_category': category,
                                     'room_category': room, 'min_pixels': 256,
                                     'consecutive_frames': 2})
            if action == 'S':
                sequence.append({'kind': 'act', 'action': 'STOP'})
            elif sequence and sequence[-1]['kind'] == 'movement' and sequence[-1]['action'] == ACTIONS[action]:
                sequence[-1]['repeat'] += 1
            else:
                sequence.append({'kind': 'movement', 'action': ACTIONS[action], 'repeat': 1})
        for order, item in enumerate(sequence):
            item['order'] = order
        return {'query_schema_version': QUERY_SCHEMA, 'coordinate_frame': 'agent_relative_discrete',
                'sequence': sequence, 'action_trace_ref': ref(trace['actions']),
                'rgb_content_refs': ['sha256:' + o['rgb_hash'] for o in trace['observations']],
                'canonicalization': 'semantic_sequence_sorted_simultaneous_events_v4'}

    def encode_query(self, query):
        """Typed data tensorization, not the future neural query encoder.

        Vocabulary is supplied in the return object, so arbitrary category sets
        cannot silently reuse incompatible integer codes across compiler objects.
        """
        if query.get('query_schema_version') != QUERY_SCHEMA or query.get('coordinate_frame') != 'agent_relative_discrete':
            raise ValueError('QUERY_SCHEMA')
        sequence = query.get('sequence')
        if not isinstance(sequence, list):
            raise ValueError('QUERY_SEQUENCE')
        out, stopped = [1, 4], False
        cats, rooms = self.query_vocabulary['categories'], self.query_vocabulary['rooms']
        for order, item in enumerate(sequence):
            if not isinstance(item, dict) or type(item.get('order')) is not int or item['order'] != order or stopped:
                raise ValueError('QUERY_ORDER_OR_STOP')
            kind = item.get('kind')
            if kind == 'movement':
                if (set(item) != {'kind', 'action', 'repeat', 'order'}
                        or item['action'] not in ('MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT')
                        or not _integer(item['repeat'], 1)):
                    raise ValueError('QUERY_MOVEMENT')
                out += [3, {'MOVE_FORWARD': 11, 'TURN_LEFT': 12, 'TURN_RIGHT': 13}[item['action']], item['repeat']]
            elif kind == 'observe':
                if (set(item) != {'kind', 'object_category', 'room_category', 'min_pixels', 'consecutive_frames', 'order'}
                        or type(item['min_pixels']) is not int or item['min_pixels'] != 256
                        or type(item['consecutive_frames']) is not int or item['consecutive_frames'] != 2
                        or (item['object_category'], item['room_category']) not in self.roles.values()):
                    raise ValueError('QUERY_EVENT')
                out += [4, cats.index(item['object_category']), rooms.index(item['room_category']), 256, 2]
            elif kind == 'act' and set(item) == {'kind', 'action', 'order'} and item['action'] == 'STOP':
                out += [5]
                stopped = True
            else:
                raise ValueError('QUERY_ACTION')
        return {'token_ids': out, 'vocabulary': dict(self.query_vocabulary),
                'encoding_schema': 'q35n.typed_query.v4'}

    def policy_at(self, trace, task_id, t, sample_id):
        if type(t) is not int or not 0 <= t < len(trace['observations']) or t > len(trace['actions']):
            raise ValueError('INVALID_POLICY_CUTOFF')
        # Check only the causal prefix; no future validity/labels enter policy.
        for j in range(t+1):
            observation = trace['observations'][j]
            if type(observation.get('step')) is not int or observation['step'] != j or not isinstance(observation.get('rgb_hash'), str):
                raise ValueError('INVALID_POLICY_OBSERVATION')
        if any(a not in ('F', 'L', 'R') for a in trace['actions'][:t]):
            raise ValueError('INVALID_EXECUTED_PREFIX')
        return {'record_type': 'policy_input', 'schema_version': 'q35n.policy_input.v4',
                'sample_id': sample_id, 'task_revision': self.task_revision,
                'task_type': 'ordered_visual_event_then_stop',
                'instruction': self.tasks[task_id]['instruction'], 'causal_cutoff_step': t,
                'observations': [{'step': j, 'rgb_ref': 'sha256:' + trace['observations'][j]['rgb_hash']}
                                 for j in range(max(0, t-1), t+1)],
                'executed_actions': [{'step': j, 'action': ACTIONS[trace['actions'][j]],
                                      'executed': True, 'collision': False}
                                     for j in range(max(0, t-8), t)],
                'memory_reset': t == 0, 'sensor_profile': json.loads(self._sensor)}

    def task_program(self, task_id):
        task = self.tasks[task_id]
        category, room = self.roles[task['anchor']]
        terminal_category, terminal_room = self.roles[task['terminal']]
        return {'anchor_object_category': category, 'anchor_room_category': room,
                'terminal_object_category': terminal_category, 'terminal_room_category': terminal_room,
                'temporal_relation': 'anchor_before_terminal_witness_then_immediate_stop'}

    def event_certificate(self, trace, role, scope):
        if scope not in ('complete_interval', 'decision_time'):
            raise ValueError('EVENT_SCOPE')
        category, room = self.roles[role]
        observations, events = trace['observations'], self.atoms(trace['observations'])
        times = range(1, len(observations)-1) if scope == 'complete_interval' else [len(observations)-1]
        matches = [(t, min(events[t][role])) for t in times if t >= 0 and events[t][role]]
        known = complete(trace)
        step, instance = matches[0] if matches and known else (None, None)
        return {'event_type': 'SEE2', 'object_category': category, 'room_category': room,
                'scope': scope, 'truth': 'T' if matches and known else 'F' if known else 'U',
                'evidence_complete': known, 'decision_step': step,
                'eligible_instance_count': len(self.eligible[role]),
                'same_instance_id': str(instance) if instance is not None else None,
                'pixel_counts': [pixel(observations[j], instance) for j in (step-1, step)] if step is not None else [],
                'rgb_refs': ['sha256:' + observations[j]['rgb_hash'] for j in (step-1, step)] if step is not None else [],
                'semantic_refs': ['sha256:' + observations[j]['semantic_hash'] for j in (step-1, step)] if step is not None else [],
                'threshold_version': 'see2_256px_2frames.v4',
                'ambiguity_codes': [] if known else ['INCOMPLETE_TRACE']}

    def action_keys(self, trace, task_id, history_hash, cutoff):
        if self.evaluate(trace, task_id) != 'pass':
            raise ValueError('ACTION_SUPERVISION_REQUIRES_PASS')
        if type(cutoff) is not int or not 0 <= cutoff < len(trace['actions']):
            raise ValueError('INVALID_ACTION_CUTOFF')
        task_hash = ref({'instruction': self.tasks[task_id]['instruction'],
                         'program': self.task_program(task_id)})
        payloads = []
        for t in range(cutoff, len(trace['actions'])):
            payloads.append({'task_hash': task_hash, 'history_hash': history_hash,
                             'decision_step': t,
                             'causal_policy_context_hash': ref(policy_semantics(self.policy_at(trace, task_id, t, 'not-encoded'))),
                             'target_continuation_lineage_hash': ref({'trace': trace['trace_hash'], 'decision': t}),
                             'target_action': ACTIONS[trace['actions'][t]],
                             'action_mask_version': 'pass_continuation_only_v1'})
        return [ref(payload) for payload in payloads], payloads
