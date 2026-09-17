"""Independent CPU-only causal loader. No model, simulator or third-party imports."""
import ast
import copy
import hashlib
import json
import random
import re
import struct
from pathlib import Path

LINE = Path(__file__).resolve().parents[3]
DATA = LINE / 'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'
ACTIONS = {'F': 'MOVE_FORWARD', 'L': 'TURN_LEFT', 'R': 'TURN_RIGHT', 'S': 'STOP'}
CATEGORY = {'chair': 20, 'sink': 21, 'bed': 22, 'tv_monitor': 23}
ROOM = {'dining room': 30, 'kitchen': 31, 'bedroom': 32, 'living room': 33}
MOVE = {'MOVE_FORWARD': 11, 'TURN_LEFT': 12, 'TURN_RIGHT': 13}
SENSOR = {'rgb_height': 224, 'rgb_width': 224, 'rgb_hfov_deg': 90,
          'sensor_position_m': [0, 1.25, 0], 'sensor_orientation_rad': [0, 0, 0],
          'recent_rgb_frames': 2, 'recent_action_window': 8}
POLICY_FIELDS = {'record_type', 'schema_version', 'sample_id', 'task_revision',
                 'task_type', 'instruction', 'causal_cutoff_step', 'observations',
                 'executed_actions', 'memory_reset', 'sensor_profile'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def ref(value):
    return 'sha256:' + sha(canonical(value))


def valid_ref(value):
    require(isinstance(value, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', value), 'INVALID_HASH_REF')


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'PATH_ESCAPE')
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_protected(root=DATA):
    root = Path(root)
    entries = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = safe_path(root, relative)
        actual = sha(path.read_bytes())
        require(actual == expected, 'SEALED_FILE_HASH:' + relative)
        entries[relative] = actual
    compiler = read_json(root / 'COMPILER_CODE_LOCK.json')
    for relative, expected in compiler.items():
        require(sha(safe_path(LINE, relative).read_bytes()) == expected, 'COMPILER_HASH:' + relative)
    return {'sealed_files_verified': len(entries), 'compiler_files_verified': len(compiler),
            'manifest_sha256': sha((root / 'SHA256SUMS').read_bytes()),
            'compiler_lock_sha256': sha((root / 'COMPILER_CODE_LOCK.json').read_bytes())}


def rgb_from_npy(blob, expected):
    """Return immutable C-order uint8 bytes; hash is raw pixels, NOT .npy bytes."""
    valid_ref(expected)
    require(blob[:6] == b'\x93NUMPY', 'NPY_MAGIC')
    version = tuple(blob[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), 'NPY_VERSION')
    size = 2 if version == (1, 0) else 4
    require(len(blob) >= 8 + size, 'NPY_TRUNCATED')
    n = struct.unpack('<H' if size == 2 else '<I', blob[8:8+size])[0]
    require(0 < n <= 65536 and len(blob) >= 8 + size + n, 'NPY_HEADER_SIZE')
    header = ast.literal_eval(blob[8+size:8+size+n].decode('utf-8' if version == (3, 0) else 'latin1'))
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'NPY_HEADER_FIELDS')
    require(header['descr'] in ('|u1', '<u1') and header['fortran_order'] is False
            and header['shape'] == (224, 224, 3), 'NPY_LAYOUT')
    pixels = blob[8+size+n:]
    require(len(pixels) == 224 * 224 * 3, 'NPY_PIXEL_LENGTH')
    require(sha(pixels) == expected[7:], 'RGB_PIXEL_HASH')
    return pixels


def validate_policy(record):
    require(set(record) == POLICY_FIELDS, 'POLICY_FIELDS')
    require(record['record_type'] == 'policy_input' and record['schema_version'] == 'q35n.policy_input.v2'
            and record['task_revision'] == 'observable_tv_sink_then_stop.v3'
            and record['task_type'] == 'ordered_visual_event_then_stop', 'POLICY_VERSION')
    require(isinstance(record['instruction'], str) and record['instruction'], 'INSTRUCTION')
    t = record['causal_cutoff_step']
    require(type(t) is int and t >= 0, 'CUTOFF')
    require(record['sensor_profile'] == SENSOR, 'SENSOR')
    require(type(record['memory_reset']) is bool and record['memory_reset'] == (t == 0), 'RESET')
    obs, acts = record['observations'], record['executed_actions']
    require(isinstance(obs, list) and all(set(o) == {'step', 'rgb_ref'} for o in obs), 'OBS_FIELDS')
    require([o['step'] for o in obs] == list(range(max(0, t-1), t+1)), 'OBS_CAUSAL_WINDOW')
    for o in obs:
        require(type(o['step']) is int, 'OBS_TIME_TYPE')
        valid_ref(o['rgb_ref'])
    require(isinstance(acts, list) and all(set(a) == {'step', 'action', 'executed', 'collision'} for a in acts), 'ACTION_FIELDS')
    require([a['step'] for a in acts] == list(range(max(0, t-8), t)), 'ACTION_CAUSAL_WINDOW')
    for a in acts:
        require(type(a['step']) is int and a['action'] in MOVE and a['executed'] is True
                and a['collision'] is False, 'EXECUTED_ACTION')


def ordered_prefix(records, expected_count):
    require(len(records) == expected_count, 'PREFIX_COUNT')
    for r in records:
        validate_policy(r)
    records = sorted(copy.deepcopy(records), key=lambda r: r['causal_cutoff_step'])
    require([r['causal_cutoff_step'] for r in records] == list(range(expected_count)), 'PREFIX_GAP_OR_DUPLICATE')
    require(len({r['instruction'] for r in records}) == 1, 'PREFIX_INSTRUCTION_CHANGED')
    observations, actions = {}, {}
    for r in records:
        for o in r['observations']:
            require(observations.setdefault(o['step'], o['rgb_ref']) == o['rgb_ref'], 'RGB_PREFIX_INCONSISTENT')
        for a in r['executed_actions']:
            require(actions.setdefault(a['step'], a['action']) == a['action'], 'ACTION_PREFIX_INCONSISTENT')
    return records


def query_projection(query):
    fields = {'query_schema_version', 'coordinate_frame', 'sequence', 'action_trace_ref',
              'rgb_content_refs', 'canonicalization'}
    require(set(query) == fields, 'QUERY_FIELDS')
    require(query['query_schema_version'] == 'q35n.continuation_query.v2'
            and query['coordinate_frame'] == 'agent_relative_discrete'
            and query['canonicalization'] == 'semantic_sequence_only_sorted_json_keys_utf8_controlled_vocab_v2', 'QUERY_VERSION')
    valid_ref(query['action_trace_ref'])
    require(isinstance(query['rgb_content_refs'], list), 'QUERY_RGB_REFS')
    for r in query['rgb_content_refs']:
        valid_ref(r)
    seq = query['sequence']
    require(isinstance(seq, list) and 2 <= len(seq) <= 160, 'QUERY_LENGTH')
    moves = 0
    for i, item in enumerate(seq):
        require(type(item.get('order')) is int and item['order'] == i, 'QUERY_ORDER')
        kind = item.get('kind')
        if kind == 'movement':
            require(set(item) == {'order', 'kind', 'action', 'repeat'}, 'MOVEMENT_FIELDS')
            require(item['action'] in MOVE and type(item['repeat']) is int
                    and 1 <= item['repeat'] <= 160, 'MOVEMENT_VALUE')
            moves += item['repeat']
        elif kind == 'observe':
            require(set(item) == {'order', 'kind', 'object_category', 'room_category', 'min_pixels', 'consecutive_frames'}, 'OBSERVE_FIELDS')
            require(item['object_category'] in CATEGORY and item['room_category'] in ROOM
                    and type(item['min_pixels']) is int and item['min_pixels'] == 256
                    and type(item['consecutive_frames']) is int and item['consecutive_frames'] == 2, 'OBSERVE_VALUE')
        else:
            require(set(item) == {'order', 'kind', 'action'} and kind == 'act'
                    and item['action'] == 'STOP' and i == len(seq)-1, 'QUERY_STOP')
    require(seq[-1]['kind'] == 'act' and moves + 1 <= 160, 'QUERY_FINAL_STOP_OR_BUDGET')
    return copy.deepcopy({k: query[k] for k in ('query_schema_version', 'coordinate_frame', 'sequence')})


def typed_query(query):
    """Typed ordered CPU tuples, NOT embedding indices for the G2 toy encoder."""
    out = []
    for item in query_projection(query)['sequence']:
        if item['kind'] == 'movement':
            out.append((3, MOVE[item['action']], item['repeat']))
        elif item['kind'] == 'observe':
            out.append((4, CATEGORY[item['object_category']], ROOM[item['room_category']], 256, 2))
        else:
            out.append((5,))
    return tuple(out)


def validate_supervision(s):
    for k in ('query_hash', 'full_log_ref', 'task_hash', 'history_hash'):
        valid_ref(s[k])
    require(ref(query_projection(s['query'])) == s['query_hash'], 'QUERY_HASH')
    arrays = [s[k] for k in ('action_targets', 'action_target_steps', 'action_loss_mask', 'action_dedup_keys')]
    require(len({len(a) for a in arrays}) == 1, 'TARGET_LENGTHS')
    steps = s['action_target_steps']
    require(all(type(t) is int and t >= s['prefix_cutoff_step'] for t in steps)
            and steps == sorted(set(steps)), 'TARGET_TIME')
    require(all(a in ACTIONS.values() for a in s['action_targets']), 'TARGET_ACTION')
    require(all(type(m) is int and m in (0, 1) for m in s['action_loss_mask']), 'TARGET_MASK')
    require(s['sample_status'] in ('labelled', 'rejected') and s['outcome'] in ('pass', 'fail', 'unknown', None), 'LABEL_STATUS')
    if s['sample_status'] == 'rejected':
        require(s['outcome'] is None, 'REJECTED_LABEL')
    if s['outcome'] != 'pass' or s['sample_status'] != 'labelled':
        require(not any(s['action_loss_mask']), 'NONPASS_ACTION_MASK')
    if s['outcome'] in ('pass', 'fail'):
        require(s['sample_status'] == 'labelled' and s['full_log_complete'] is True
                and s['physical_merge_certificate']['truth'] == 'T' and not s['generation_failures']
                and s['synthetic_spec_only'] is False, 'LABEL_WITHOUT_EVIDENCE')


def supervision_label(s):
    validate_supervision(s)
    known = s['sample_status'] == 'labelled' and s['outcome'] in ('pass', 'fail')
    return {'target': int(s['outcome'] == 'pass') if known else None, 'bce_mask': int(known)}


class FamilyLoader:
    def __init__(self, root=DATA, verify=True):
        self.root = Path(root).resolve()
        self.protection = verify_protected(self.root) if verify else None
        self.manifest = read_json(self.root / 'FAMILY_MANIFEST.json')
        require(self.manifest['split'] == 'interface_only' and not self.manifest['synthetic_spec_only'], 'DATA_SCOPE')
        self.index = read_json(self.root / 'POLICY_PREFIX_INDEX.json')
        self.supervision = read_jsonl(self.root / 'SUPERVISION_ONLY.jsonl')
        self.dedup = read_json(self.root / 'ACTION_DEDUP.json')
        self.current = {r['sample_id']: r for r in read_jsonl(self.root / 'POLICY_INPUT.jsonl')}
        self.prefixes = {}
        self._pixels = {}
        self._traces = {}
        for item in self.index:
            key = (item['task_id'], item['history_id'])
            require(key not in self.prefixes and item['requires_recurrent_replay_from_zero'] is True, 'PREFIX_GROUP')
            self.prefixes[key] = ordered_prefix(read_jsonl(safe_path(self.root, item['path'])), item['decision_count'])
        expected = {(t, h, c) for t in self.manifest['task_ids'] for h in self.manifest['history_ids']
                    for c in self.manifest['continuation_trace_ids']}
        require(len(self.supervision) == len(expected) and
                {(s['task_id'], s['history_id'], s['continuation_trace_id']) for s in self.supervision} == expected, 'CROSS_MATRIX')
        require(set(self.prefixes) == {(t, h) for t, h, _ in expected}, 'PREFIX_MATRIX')
        require(len(self.current) == len(self.supervision), 'CURRENT_MATRIX')
        self.by_id = {}
        for s in self.supervision:
            validate_supervision(s)
            require(s['family_id'] == self.manifest['family_id'] and s['house_id'] == self.manifest['house_id']
                    and s['split'] == self.manifest['split'], 'GROUP_METADATA')
            require(s['sample_id'] not in self.by_id, 'DUPLICATE_SAMPLE')
            self.by_id[s['sample_id']] = s
            r = self.current[s['sample_id']]
            validate_policy(r)
            prefix_end = self.prefixes[(s['task_id'], s['history_id'])][-1]
            require(s['prefix_cutoff_step'] == prefix_end['causal_cutoff_step'], 'PREFIX_CUTOFF')
            require(self.context(r) == self.context(prefix_end), 'CURRENT_PREFIX_MISMATCH')
        self.validate_owners()

    @staticmethod
    def context(record):
        return {k: v for k, v in record.items() if k not in ('sample_id', 'record_type', 'schema_version')}

    def policy_payload(self, record):
        validate_policy(record)
        rgb = []
        for o in record['observations']:
            key = o['rgb_ref']
            if key not in self._pixels:
                path = safe_path(self.root, 'content/' + key[7:] + '.rgb.npy')
                self._pixels[key] = rgb_from_npy(path.read_bytes(), key)
            rgb.append(self._pixels[key])
        return {'instruction': record['instruction'], 'rgb': tuple(rgb),
                'executed_actions': tuple(a['action'] for a in record['executed_actions']),
                'memory_reset': record['memory_reset']}

    def prefix_groups(self):
        return tuple(sorted(self.prefixes))

    def iter_prefix(self, task_id, history_id):
        for r in self.prefixes[(task_id, history_id)]:
            yield {'control': {'decision_step': r['causal_cutoff_step']},
                   'policy': self.policy_payload(r)}

    def query_input(self, sample_id):
        return typed_query(self.by_id[sample_id]['query'])

    def label(self, sample_id):
        return supervision_label(self.by_id[sample_id])

    def grouped_cells(self):
        result = {}
        for s in self.supervision:
            if self.label(s['sample_id'])['bce_mask']:
                result.setdefault(s['family_id'], {}).setdefault(s['task_id'], []).append(s['sample_id'])
        return {f: {t: tuple(sorted(cells)) for t, cells in sorted(tasks.items())}
                for f, tasks in sorted(result.items())}

    def trace(self, s):
        # Private compiler-side metadata; NEVER yielded to the policy caller.
        key = (s['history_id'], s['continuation_trace_id'])
        if key not in self._traces:
            path = safe_path(self.root, 'physical_traces/1109_%s_%s.json' % key)
            tr = read_json(path)
            require(tr['complete'] is True and tr['collisions'] == 0 and tr['actions'][-1] == 'S'
                    and 'S' not in tr['actions'][:-1] and len(tr['actions']) == len(tr['observations']), 'TRACE_STRUCTURE')
            require([o['step'] for o in tr['observations']] == list(range(len(tr['observations']))), 'TRACE_ORDER')
            self._traces[key] = tr
        tr = self._traces[key]
        require('sha256:' + tr['trace_hash'] == s['full_log_ref'], 'TRACE_JOIN_HASH')
        return tr

    def decision_record(self, s, t):
        tr = self.trace(s)
        require(type(t) is int and 0 <= t < len(tr['observations']), 'DECISION_RANGE')
        r = copy.deepcopy(self.current[s['sample_id']])
        r['causal_cutoff_step'] = t
        r['memory_reset'] = t == 0
        r['observations'] = [{'step': i, 'rgb_ref': 'sha256:' + tr['observations'][i]['rgb_hash']}
                             for i in range(max(0, t-1), t+1)]
        r['executed_actions'] = [{'step': i, 'action': ACTIONS[tr['actions'][i]], 'executed': True, 'collision': False}
                                for i in range(max(0, t-8), t)]
        validate_policy(r)
        return r

    def validate_owners(self):
        actual = {}
        for s in self.supervision:
            tr = self.trace(s)
            cutoff = s['prefix_cutoff_step']
            require(s['action_target_steps'] == list(range(cutoff, len(tr['actions']))), 'TARGET_TRACE_TIME')
            for i, (key, step, action, mask) in enumerate(zip(s['action_dedup_keys'], s['action_target_steps'],
                                                           s['action_targets'], s['action_loss_mask'])):
                valid_ref(key)
                payload = self.dedup['payloads'][key]
                require(ref(payload) == key, 'DEDUP_PAYLOAD_HASH')
                require(payload['task_hash'] == s['task_hash'] and payload['history_hash'] == s['history_hash']
                        and payload['decision_step'] == step and payload['target_action'] == action
                        and action == ACTIONS[tr['actions'][step]], 'DEDUP_LINEAGE')
                r = self.decision_record(s, step)
                require(ref(self.context(r)) == payload['causal_policy_context_hash'], 'DEDUP_CAUSAL_CONTEXT')
                require(ref({'trace': tr['trace_hash'], 'decision': step}) == payload['target_continuation_lineage_hash'], 'DEDUP_TRACE_LINEAGE')
                if mask:
                    require(key not in actual, 'DUPLICATE_CE_OWNER')
                    actual[key] = [s['sample_id'], i]
        require(actual == self.dedup['owners'], 'CE_OWNER_MISMATCH')

    def iter_action_stream(self, sample_id):
        """From zero through each actual decision. Targets/masks stay separate.

        Each positive continuation must restart/replay its memory (or use an
        explicitly isolated snapshot). No standalone late target can warm-start
        from another cell's memory. Failed cells yield no action loss.
        """
        s = self.by_id[sample_id]
        tr = self.trace(s)
        cutoff = s['prefix_cutoff_step']
        for t in range(len(tr['observations'])):
            target = {'action': None, 'ce_mask': 0}
            if t >= cutoff:
                i = t - cutoff
                target = {'action': s['action_targets'][i], 'ce_mask': s['action_loss_mask'][i]}
            yield {'control': {'decision_step': t}, 'policy': self.policy_payload(self.decision_record(s, t)),
                   'action_supervision': target}


def sample_grouped(loaders, count, seed):
    """Uniform family then task then valid cell; excludes seeds as samples."""
    groups = {}
    for loader in loaders:
        for family, tasks in loader.grouped_cells().items():
            require(family not in groups, 'DUPLICATE_FAMILY')
            groups[family] = tasks
    require(groups and type(count) is int and count >= 0, 'SAMPLING_INPUT')
    rng = random.Random(seed)
    for _ in range(count):
        f = rng.choice(sorted(groups))
        t = rng.choice(sorted(groups[f]))
        yield (f, t, rng.choice(groups[f][t]))
