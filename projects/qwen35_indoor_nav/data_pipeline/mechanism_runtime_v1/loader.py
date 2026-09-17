"""V4 CPU readback and explicit policy/teacher boundary; no model imports."""
import ast
import copy
import hashlib
import json
import struct
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]
POLICY_FIELDS = {'record_type', 'schema_version', 'sample_id', 'task_revision',
                 'task_type', 'instruction', 'causal_cutoff_step', 'observations',
                 'executed_actions', 'memory_reset', 'sensor_profile'}
MOVES = {'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'RELATIVE_PATH_REQUIRED')
    result = (Path(root).resolve() / relative).resolve()
    require(result.is_relative_to(Path(root).resolve()), 'PATH_ESCAPE')
    return result


def npy_pixels(blob, expected, kind):
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'PIXEL_HASH_FORMAT')
    require(blob[:6] == b'\x93NUMPY' and tuple(blob[6:8]) in ((1, 0), (2, 0), (3, 0)), 'NPY_HEADER')
    size = 2 if blob[6] == 1 else 4
    require(len(blob) >= 8 + size, 'NPY_TRUNCATED')
    length = struct.unpack('<H' if size == 2 else '<I', blob[8:8+size])[0]
    require(0 < length <= 65536 and len(blob) >= 8+size+length, 'NPY_HEADER_LENGTH')
    header = ast.literal_eval(blob[8+size:8+size+length].decode('utf8' if blob[6] == 3 else 'latin1'))
    require(set(header) == {'descr', 'fortran_order', 'shape'} and header['fortran_order'] is False, 'NPY_LAYOUT')
    shape, descr, nbytes = ((224, 224, 3), ('|u1', '<u1'), 224*224*3) if kind == 'rgb' else ((224, 224), ('<u4',), 224*224*4)
    require(kind in ('rgb', 'semantic') and header['shape'] == shape and header['descr'] in descr, 'NPY_DTYPE_SHAPE')
    pixels = blob[8+size+length:]
    require(len(pixels) == nbytes and sha(pixels) == expected, 'PIXEL_HASH_OR_LENGTH')
    return pixels


def validate_policy(record):
    require(set(record) == POLICY_FIELDS, 'POLICY_FIELDS')
    require(record['record_type'] == 'policy_input' and record['schema_version'] == 'q35n.policy_input.v4', 'POLICY_VERSION')
    require(isinstance(record['instruction'], str) and bool(record['instruction']), 'INSTRUCTION')
    t = record['causal_cutoff_step']
    require(type(t) is int and t >= 0, 'CUTOFF')
    require(type(record['memory_reset']) is bool and record['memory_reset'] == (t == 0), 'RESET')
    observations, actions = record['observations'], record['executed_actions']
    require(isinstance(observations, list) and all(set(x) == {'step', 'rgb_ref'} for x in observations), 'OBS_FIELDS')
    require([o['step'] for o in observations] == list(range(max(0, t-1), t+1)), 'OBS_CAUSAL_WINDOW')
    require(isinstance(actions, list) and all(set(a) == {'step', 'action', 'executed', 'collision'} for a in actions), 'ACTION_FIELDS')
    require([a['step'] for a in actions] == list(range(max(0, t-8), t)), 'ACTION_CAUSAL_WINDOW')
    require(all(type(o['step']) is int and isinstance(o['rgb_ref'], str) and o['rgb_ref'].startswith('sha256:')
                and len(o['rgb_ref']) == 71 and all(c in '0123456789abcdef' for c in o['rgb_ref'][7:]) for o in observations), 'OBS_REF')
    require(all(type(a['step']) is int and a['action'] in MOVES and a['executed'] is True
                and a['collision'] is False for a in actions), 'ACTION_VALUES')
    expected_sensor = {'rgb_height': 224, 'rgb_width': 224, 'rgb_hfov_deg': 90,
                       'sensor_position_m': [0, 1.25, 0], 'sensor_orientation_rad': [0, 0, 0],
                       'recent_rgb_frames': 2, 'recent_action_window': 8}
    require(record['sensor_profile'] == expected_sensor and record['task_type'] == 'ordered_visual_event_then_stop', 'SENSOR_TASK')


class FamilyLoader:
    def __init__(self, root, compiler):
        self.root = Path(root).resolve()
        require(self.root.is_relative_to(LINE), 'EXPORT_OUTSIDE_LINE')
        self.compiler = compiler
        self.manifest = self.read('MANIFEST.json')
        require(self.manifest['schema_version'] == 'q35n.family_export.v4', 'EXPORT_VERSION')
        config = {'roles': {k: list(v) for k, v in compiler.roles.items()},
                  'eligible': {k: list(v) for k, v in compiler.eligible.items()},
                  'tasks': {k: dict(v) for k, v in compiler.tasks.items()},
                  'task_revision': compiler.task_revision}
        require(self.manifest['compiler_config'] == config, 'COMPILER_CONFIG_MISMATCH')
        require(self.manifest['scientific_pass'] is False and self.manifest['training_admission'] is False, 'UNAPPROVED_ADMISSION')
        self.contents = self.read('CONTENT_INDEX.json')
        self.prefix_index = self.read('PREFIX_INDEX.json')
        self.cells = self.lines('SUPERVISION_ONLY.jsonl')
        self.owners = self.read('ACTION_OWNERS.json')
        self._rgb_cache = {}
        for line in (self.root / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            require(sha(safe_path(self.root, relative).read_bytes()) == expected, 'EXPORT_FILE_HASH:' + relative)

    def read(self, relative):
        return json.loads(safe_path(self.root, relative).read_text())

    def lines(self, relative):
        return [json.loads(x) for x in safe_path(self.root, relative).read_text().splitlines() if x]

    def rgb(self, reference):
        key = reference[7:] + '.rgb'
        item = self.contents[key]
        blob = safe_path(LINE, item['line_relative_path']).read_bytes()
        require(sha(blob) == item['file_sha256'], 'CONTENT_FILE_CHANGED')
        return npy_pixels(blob, reference[7:], 'rgb')

    def policy_payload(self, record):
        """Model input only. No ID, hash, future query, label, pose or semantic pixels."""
        validate_policy(record)
        return {'task_type': record['task_type'], 'instruction': record['instruction'],
                'rgb': [self.rgb(o['rgb_ref']) for o in record['observations']],
                'executed_actions': [a['action'] for a in record['executed_actions']],
                'memory_reset': record['memory_reset']}

    def prefix_records(self, prefix_id):
        item = self.prefix_index[prefix_id]
        records = self.lines(item['path'])
        require(len(records) == item['decision_count'], 'PREFIX_COUNT')
        seen_rgb, seen_actions = {}, {}
        for t, record in enumerate(records):
            validate_policy(record)
            require(record['causal_cutoff_step'] == t, 'PREFIX_GAP_OR_REORDER')
            require(record['instruction'] == records[0]['instruction'], 'PREFIX_TASK_CHANGED')
            for o in record['observations']:
                require(seen_rgb.setdefault(o['step'], o['rgb_ref']) == o['rgb_ref'], 'PREFIX_RGB_CONFLICT')
            for a in record['executed_actions']:
                require(seen_actions.setdefault(a['step'], a['action']) == a['action'], 'PREFIX_ACTION_CONFLICT')
        return records

    def supervision(self, cell_id):
        """Reader-only target; caller must never append this to policy inputs."""
        cell = next(c for c in self.cells if c['cell_id'] == cell_id)
        require(cell['outcome'] in ('pass', 'fail', 'unknown'), 'OUTCOME')
        y = {'pass': 1, 'fail': 0, 'unknown': None}[cell['outcome']]
        require(cell['y'] == y and cell['bce_mask'] == int(y is not None), 'UNKNOWN_IS_NOT_NEGATIVE')
        encoded = self.compiler.encode_query(cell['query'])
        return {'query': copy.deepcopy(encoded), 'y': y, 'bce_mask': cell['bce_mask'],
                'm2_program_state': copy.deepcopy(cell['m2_program_state']),
                'action_targets': copy.deepcopy(cell['action_targets']),
                'action_loss_mask': copy.deepcopy(cell['action_loss_mask'])}

    def action_stream(self, cell_id):
        cell = next(c for c in self.cells if c['cell_id'] == cell_id)
        records = self.lines(cell['full_policy_path'])
        require([r['causal_cutoff_step'] for r in records] == list(range(len(records))), 'FULL_STREAM_GAP')
        for record in records:
            validate_policy(record)
        return records

    def validate_supervision_contract(self):
        """Independently re-evaluate every label/state and owner from exported logs."""
        owner_actual, payload_actual = {}, {}
        outcomes = {'pass': 0, 'fail': 0, 'unknown': 0}
        require(len(self.cells) == self.manifest['cells'] == 18, 'CELL_COUNT')
        require(len({x['cell_id'] for x in self.cells}) == 18, 'CELL_IDS')
        for cell in self.cells:
            trace = self.read(cell['trace_path'])
            task, cutoff = cell['task_id'], cell['prefix_cutoff']
            outcome = self.compiler.evaluate(trace, task)
            require(cell['outcome'] == outcome, 'LABEL_RECOMPUTATION')
            outcomes[outcome] += 1
            self.supervision(cell['cell_id'])
            require(cell['query'] == self.compiler.query_from_trace(self.compiler.slice_continuation(trace, cutoff)), 'QUERY_RECOMPUTATION')
            require(cell['m2_program_state'] == self.compiler.m2(trace, task), 'M2_RECOMPUTATION')
            # Canonical keys duplicate compiler digest semantics for JSON-loaded logs.
            canonical = json.dumps(trace['observations'][:cutoff+1], sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()
            history_hash = 'sha256:' + sha(canonical)
            keys, values = self.compiler.action_keys(trace, task, history_hash, cutoff) if outcome == 'pass' else ([], [])
            require(cell['action_keys'] == keys and cell['action_targets'] == [v['target_action'] for v in values]
                    and cell['action_steps'] == [v['decision_step'] for v in values], 'CE_RECOMPUTATION')
            require(len(cell['action_loss_mask']) == len(keys), 'CE_MASK_LENGTH')
            for i, (key, value, mask) in enumerate(zip(keys, values, cell['action_loss_mask'])):
                require(type(mask) is int and mask == int(key not in owner_actual), 'CE_OWNER_MASK')
                require(payload_actual.setdefault(key, value) == value, 'CE_PAYLOAD_CONFLICT')
                owner_actual.setdefault(key, [cell['cell_id'], i])
        require(self.owners == {'owners': owner_actual, 'payloads': payload_actual}, 'CE_OWNER_INDEX')
        require(len(owner_actual) == self.manifest['ce_unique_owners'], 'CE_OWNER_COUNT')
        require({k: v for k, v in outcomes.items() if v} == self.manifest['outcomes'], 'OUTCOME_COUNTS')
        return {'labels_recomputed': len(self.cells), 'ce_owners_verified': len(owner_actual), 'outcomes': outcomes}
