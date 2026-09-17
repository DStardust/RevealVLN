"""Read-only ordinary production adapter; importing needs only the stdlib.

Only decision()['policy'] is a decoded model input. Metadata paths stay in the
control channel. This adapter neither grants training nor changes source files.
"""
import hashlib
import json
from pathlib import Path
import re

LINE = Path(__file__).resolve().parents[2]
ROOT = LINE.parents[1]
ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')
ACTION_MAP = {a: a for a in ACTIONS}
ACTION_MAP.update(MOVE_FORWARD='move_forward', TURN_LEFT='turn_left', TURN_RIGHT='turn_right')
ROW_FIELDS = frozenset(('sourceRoot', 'policy_file', 'supervision_file',
                        'rgb_reference_root', 'source', 'scene_group',
                        'physical_source_route_sha256', 'decisions'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _relative(base, value, *, directory=False):
    require(isinstance(value, str) and bool(value), 'RELATIVE_PATH_REQUIRED')
    part = Path(value)
    require(not part.is_absolute() and '..' not in part.parts, 'PATH_ESCAPE')
    result = (base / part).resolve()
    require(result.is_relative_to(base.resolve()), 'PATH_ESCAPE')
    require(result.is_dir() if directory else result.is_file(), 'MISSING_PATH')
    return result


def normalize_action(action):
    require(isinstance(action, str) and action in ACTION_MAP, 'INVALID_ACTION')
    return ACTION_MAP[action]


def validate_row(row):
    """Validate metadata and resolve ROOT-relative sourceRoot; read no pixels."""
    require(isinstance(row, dict) and ROW_FIELDS <= row.keys(), 'ROW_FIELDS')
    for name in ('source', 'scene_group'):
        require(isinstance(row[name], str) and bool(row[name]), 'ROW_' + name)
    require(type(row['decisions']) is int and row['decisions'] > 0, 'ROW_DECISIONS')
    require(isinstance(row['physical_source_route_sha256'], str) and
            re.fullmatch('[0-9a-f]{64}', row['physical_source_route_sha256']) is not None,
            'ROW_PHYSICAL_HASH')
    source = _relative(ROOT, row['sourceRoot'], directory=True)
    require(source.is_relative_to(LINE) and source != LINE, 'SOURCE_OUTSIDE_LINE')
    return source


def load_rgb(path):
    """Decode on demand; verify raw RGB bytes, not compressed PNG bytes."""
    from PIL import Image
    path = Path(path)
    require(path.suffix == '.png' and re.fullmatch('[0-9a-f]{64}', path.stem)
            is not None, 'RGB_REFERENCE_FORMAT')
    with Image.open(path) as source:
        require(source.format == 'PNG' and source.mode == 'RGB'
                and source.size == (224, 224), 'RGB_FORMAT_SHAPE')
        source.load()
        image = source.copy()
    require(hashlib.sha256(image.tobytes()).hexdigest() == path.stem, 'RGB_RAW_PIXEL_HASH')
    return image


class OrdinaryRecord:
    """One instruction-conditioned stream with lazy RGB decoding.

    Each iteration starts at zero. The runner must honor memory_reset and order.
    No future frames, simulator metadata, or target actions enter policy payloads.
    """
    def __init__(self, row):
        self.source_root = validate_row(row)
        self._row = {key: row[key] for key in ROW_FIELDS}
        self.policy_path = _relative(self.source_root, row['policy_file'])
        self.supervision_path = _relative(self.source_root, row['supervision_file'])
        self.rgb_root = _relative(self.source_root, row['rgb_reference_root'], directory=True)
        policy_blob = self.policy_path.read_bytes()
        supervision_blob = self.supervision_path.read_bytes()
        for field, blob in (('policy_sha256', policy_blob), ('supervision_sha256', supervision_blob)):
            if field in row:
                require(isinstance(row[field], str) and
                        hashlib.sha256(blob).hexdigest() == row[field], 'SOURCE_JSON_HASH:' + field)
        policy = json.loads(policy_blob)
        supervision = json.loads(supervision_blob)
        require(isinstance(policy, dict) and set(policy) == {'instruction', 'rgb_sequence'}, 'POLICY_FIELDS')
        require(isinstance(policy['instruction'], str) and bool(policy['instruction'].strip()), 'INSTRUCTION')
        require(isinstance(policy['rgb_sequence'], list), 'RGB_SEQUENCE')
        require(isinstance(supervision, dict) and isinstance(supervision.get('actions'), list), 'ACTION_SEQUENCE')
        self.instruction = policy['instruction']
        self.actions = tuple(normalize_action(a) for a in supervision['actions'])
        self._refs = tuple(policy['rgb_sequence'])
        require(len(self.actions) == len(self._refs) == row['decisions'], 'LENGTH_MISMATCH')
        require(self.actions[-1] == 'STOP' and 'STOP' not in self.actions[:-1], 'TERMINAL_STOP')
        # Paths checked, no PNG decoded and no full-dataset pixel cache created.
        for ref in self._refs:
            path = _relative(self.rgb_root, ref)
            require(path.suffix == '.png' and re.fullmatch('[0-9a-f]{64}', path.stem)
                    is not None, 'RGB_REFERENCE_FORMAT')

    def __len__(self):
        return len(self.actions)

    def metadata(self):
        return dict(self._row, action_counts={a: self.actions.count(a) for a in ACTIONS},
                    terminal_stop_valid=True, pixel_content_verified=False)

    def decision_metadata(self, t):
        """CPU causal projection; rgb_refs are control data, not model inputs."""
        require(type(t) is int and 0 <= t < len(self), 'DECISION_TIME')
        return {'control': {'decision_step': t, 'memory_reset': t == 0,
                            'rgb_refs': list(self._refs[max(0, t-1):t+1])},
                'policy': {'instruction': self.instruction,
                           'executed_actions': list(self.actions[max(0, t-8):t])},
                'supervision': {'target_action': self.actions[t], 'ce_mask': 1}}

    def decision(self, t, decode_rgb=True):
        """Decoded policy whitelist plus separately returned reset and target."""
        result = self.decision_metadata(t)
        require(type(decode_rgb) is bool, 'DECODE_FLAG')
        if not decode_rgb:
            return result
        refs = result['control'].pop('rgb_refs')
        result['policy']['images'] = [load_rgb(_relative(self.rgb_root, ref)) for ref in refs]
        require(set(result['policy']) == {'instruction', 'images', 'executed_actions'}, 'MODEL_WHITELIST')
        return result

    def iter_decisions(self, *, decode=False):
        reader = self.decision if decode else self.decision_metadata
        for t in range(len(self)):
            yield reader(t)
