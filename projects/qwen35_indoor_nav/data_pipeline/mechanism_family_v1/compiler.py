"""Pure, independently recomputed event/query/causal supervision operations.

All metadata functions are compiler-side only. Deployment receives the explicit
policy whitelist; not an arbitrary filtered copy of a supervision record.
"""
import hashlib
import json


ACTIONS = {'F': 'MOVE_FORWARD', 'L': 'TURN_LEFT', 'R': 'TURN_RIGHT', 'S': 'STOP'}
KINDS = {'D': ('chair', 'dining room'), 'K': ('sink', 'kitchen'),
         'B': ('bed', 'bedroom'), 'L': ('tv_monitor', 'living room')}
TASKS = {
    'g_D_v2': '先连续两帧看见餐厅内的一把椅子，再连续两帧看见卧室内的一张床，然后停止。',
    'g_K_v2': '先连续两帧看见厨房内的水槽，再连续两帧看见卧室内的一张床，然后停止。'}
TASK_ANCHOR = {'g_D_v2': 'D', 'g_K_v2': 'K'}
TASK_REVISION = 'observable_see2_then_stop.v2'
SENSOR = {'rgb_height': 224, 'rgb_width': 224, 'rgb_hfov_deg': 90,
          'sensor_position_m': [0, 1.25, 0], 'sensor_orientation_rad': [0, 0, 0],
          'recent_rgb_frames': 2, 'recent_action_window': 8}


def canonical(x):
    def keys(value):
        if isinstance(value, dict):
            if len({str(k) for k in value}) != len(value): raise ValueError('JSON_KEY_COLLISION')
            return {str(k): keys(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [keys(v) for v in value]
        return value
    return json.dumps(keys(x), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else canonical(x)).hexdigest()


def ref(x):
    return 'sha256:' + digest(x)


def pixel(o, idx):
    return o['pixels'].get(str(idx), o['pixels'].get(idx, 0))


def atoms(observations, eligible):
    """Recompute from counts, never trust stored see2 booleans."""
    output = []
    for t, o in enumerate(observations):
        if t == 0:
            output.append({kind: [] for kind in KINDS})
        elif not o['evidence_complete'] or not observations[t-1]['evidence_complete']:
            output.append({kind: None for kind in KINDS})
        else:
            output.append({kind: [idx for idx in ids if pixel(o, idx) >= 256 and pixel(observations[t-1], idx) >= 256]
                           for kind, ids in eligible.items()})
    return output


def complete(trace):
    acts, obs = trace['actions'], trace['observations']
    return (bool(trace['complete']) and trace['collisions'] == 0
            and all(a in ACTIONS for a in acts)
            and 'S' not in acts[:-1]
            and len(obs) == len(acts) + (0 if acts and acts[-1] == 'S' else 1)
            and [o['step'] for o in obs] == list(range(len(obs)))
            and all(o['evidence_complete'] for o in obs))


def evaluate(trace, task, eligible):
    if not complete(trace):
        return 'unknown'
    ev = atoms(trace['observations'], eligible)
    anchor = TASK_ANCHOR[task]
    return 'pass' if trace['actions'][-1:] == ['S'] and ev[-1]['B'] and any(e[anchor] for e in ev[:-1]) else 'fail'


def m2(trace, task, eligible):
    ev = atoms(trace['observations'], eligible)
    anchor = TASK_ANCHOR[task]
    result, seen, known = [], False, True
    for t, e in enumerate(ev):
        known = known and trace['observations'][t]['evidence_complete']
        # strict t_anchor < t_terminal: a simultaneous first witness is not ready.
        ready = seen and bool(e['B'])
        seen = seen or bool(e[anchor])
        state = 'UNKNOWN' if not known else ('READY_TO_STOP' if ready else 'WAIT_TERMINAL_WITNESS' if seen else 'WAIT_ANCHOR')
        result.append({'decision_step': t, 'state': state, 'truth_known': known,
                       'loss_mask': int(known), 'causal_event_cutoff_step': t})
    return result


def query_from_trace(trace, eligible):
    if not complete(trace):
        raise ValueError('INCOMPLETE_QUERY_TRACE')
    ev = atoms(trace['observations'], eligible)
    seq = []
    for t, action in enumerate(trace['actions']):
        if t:
            for kind in KINDS:
                if ev[t][kind]:
                    cat, room = KINDS[kind]
                    seq.append({'kind': 'observe', 'object_category': cat, 'room_category': room,
                                'min_pixels': 256, 'consecutive_frames': 2})
        if action == 'S':
            seq.append({'kind': 'act', 'action': 'STOP'})
        elif seq and seq[-1]['kind'] == 'movement' and seq[-1]['action'] == ACTIONS[action]:
            seq[-1]['repeat'] += 1
        else:
            seq.append({'kind': 'movement', 'action': ACTIONS[action], 'repeat': 1})
    for i, item in enumerate(seq):
        item['order'] = i
    return {'query_schema_version': 'q35n.continuation_query.v2', 'coordinate_frame': 'agent_relative_discrete',
            'sequence': seq, 'action_trace_ref': ref(trace['actions']),
            'rgb_content_refs': ['sha256:' + o['rgb_hash'] for o in trace['observations']],
            'canonicalization': 'semantic_sequence_only_sorted_json_keys_utf8_controlled_vocab_v2'}


def semantic_query(query):
    return {k: query[k] for k in ('query_schema_version', 'coordinate_frame', 'sequence')}


def encode_query(query):
    """Deterministic controlled-vocabulary integer sequence; no hash/ID features.

    This is a data-interface tensorization, not a trained neural query encoder.
    Each event has fixed categorical codes; variable movement counts are numeric.
    """
    out = [1, 2]  # schema v2, agent-relative discrete
    cats = {'chair': 20, 'sink': 21, 'bed': 22, 'tv_monitor': 23}
    rooms = {'dining room': 30, 'kitchen': 31, 'bedroom': 32, 'living room': 33}
    codes = {'MOVE_FORWARD': 11, 'TURN_LEFT': 12, 'TURN_RIGHT': 13}
    for i, item in enumerate(query['sequence']):
        if item['order'] != i:
            raise ValueError('QUERY_ORDER')
        if item['kind'] == 'movement':
            out += [3, codes[item['action']], item['repeat']]
        elif item['kind'] == 'observe':
            out += [4, cats[item['object_category']], rooms[item['room_category']], 256, 2]
        else:
            if item['kind'] != 'act' or item['action'] != 'STOP':
                raise ValueError('QUERY_ACTION')
            out += [5]
    return out


def policy_at(trace, task, t, sample_id):
    assert 0 <= t < len(trace['observations'])
    return {'record_type': 'policy_input', 'schema_version': 'q35n.policy_input.v2',
            'sample_id': sample_id, 'task_revision': TASK_REVISION,
            'task_type': 'ordered_visual_event_then_stop', 'instruction': TASKS[task],
            'causal_cutoff_step': t,
            'observations': [{'step': j, 'rgb_ref': 'sha256:' + trace['observations'][j]['rgb_hash']} for j in range(max(0, t-1), t+1)],
            'executed_actions': [{'step': j, 'action': ACTIONS[trace['actions'][j]], 'executed': True, 'collision': False}
                                 for j in range(max(0, t-8), t)],
            'memory_reset': t == 0, 'sensor_profile': dict(SENSOR)}


def policy_semantics(record):
    return {k: v for k, v in record.items() if k not in ('sample_id', 'record_type', 'schema_version')}


def event_certificate(trace, kind, eligible, scope):
    obs = trace['observations']; ev = atoms(obs, eligible)
    ts = range(1, len(obs)-1) if scope == 'complete_interval' else [len(obs)-1]
    matches = [(t, min(ev[t][kind])) for t in ts if ev[t][kind]]
    known = complete(trace)
    truth = 'T' if matches and known else 'F' if known else 'U'
    t, idx = matches[0] if matches and known else (None, None)
    return {'event_type': 'SEE2', 'object_category': KINDS[kind][0], 'room_category': KINDS[kind][1],
            'scope': scope, 'truth': truth, 'evidence_complete': known, 'decision_step': t,
            'eligible_instance_count': len(eligible[kind]), 'same_instance_id': str(idx) if idx is not None else None,
            'pixel_counts': [pixel(obs[j], idx) for j in (t-1, t)] if t is not None else [],
            'rgb_refs': ['sha256:' + obs[j]['rgb_hash'] for j in (t-1, t)] if t is not None else [],
            'semantic_refs': ['sha256:' + obs[j]['semantic_hash'] for j in (t-1, t)] if t is not None else [],
            'threshold_version': 'see2_256px_2frames.v2', 'ambiguity_codes': [] if known else ['INCOMPLETE_TRACE']}


def slice_continuation(trace, cutoff):
    out = {'actions': trace['actions'][cutoff:],
           'observations': [dict(o, step=i) for i, o in enumerate(trace['observations'][cutoff:])],
           'collisions': trace['collisions'], 'complete': trace['complete']}
    return out


def task_program(task):
    cat, room = KINDS[TASK_ANCHOR[task]]
    return {'anchor_object_category': cat, 'anchor_room_category': room,
            'terminal_object_category': 'bed', 'terminal_room_category': 'bedroom',
            'temporal_relation': 'anchor_before_terminal_witness_then_immediate_stop'}


def action_keys(trace, task, history_hash, cutoff):
    keys, payloads = [], []
    th = ref({'instruction': TASKS[task], 'program': task_program(task)})
    for t in range(cutoff, len(trace['actions'])):
        # Entire demonstration lineage is supervision-only, never policy input.
        payload = {'task_hash': th, 'history_hash': history_hash, 'decision_step': t,
                   'causal_policy_context_hash': ref(policy_semantics(policy_at(trace, task, t, 'not-encoded'))),
                   'target_continuation_lineage_hash': ref({'trace': trace['trace_hash'], 'decision': t}),
                   'target_action': ACTIONS[trace['actions'][t]], 'action_mask_version': 'pass_continuation_only_v1'}
        keys.append(ref(payload)); payloads.append(payload)
    return keys, payloads
