"""No framework imports: bounded IO, exact action map and policy message contract."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
TRAIN = LINE / 'sft_acceptance/ordinary_sync_recovery_v1'
ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')
HABITAT_IDS = (1, 2, 3, 0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, obj, exclusive=False):
    path = Path(path)
    blob = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x') as f:
            f.write(blob)
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(blob)
        tmp.replace(path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')


def advance(action, count, done, limit=500):
    if action not in ACTIONS or done or count >= limit:
        raise ValueError('INVALID_STEP_OR_AFTER_DONE')
    count += 1
    return count, action == 'STOP' or count >= limit


def xyzw_to_wxyz(value):
    if len(value) != 4:
        raise ValueError('QUATERNION_LENGTH')
    x, y, z, w = value
    return w, x, y, z


class Window:
    def __init__(self):
        self.instruction, self.images, self.executed = None, [], []

    def receive(self, payload, executed=None):
        if payload.get('done') is True:
            if set(payload) != {'done'}:
                raise ValueError('PRIVILEGED_TERMINAL_FIELDS')
            return False
        expected = {'done', 'rgb', 'instruction'} if executed is None else {'done', 'rgb'}
        if set(payload) != expected or payload['done'] is not False:
            raise ValueError('POLICY_WHITELIST')
        blob = base64.b64decode(payload['rgb'], validate=True)
        if len(blob) != 224*224*3:
            raise ValueError('RGB_BYTES')
        if executed is None:
            if not isinstance(payload['instruction'], str) or not payload['instruction'].strip():
                raise ValueError('INSTRUCTION')
            self.instruction = payload['instruction']
            self.images, self.executed = [], []
        else:
            if executed not in ACTIONS[:-1] or self.instruction is None:
                raise ValueError('EXECUTED_ACTION')
            self.executed = (self.executed + [executed])[-8:]
        self.images = (self.images + [blob])[-2:]
        return True

    def item(self):
        from PIL import Image
        return dict(instruction=self.instruction,
                    images=[Image.frombytes('RGB', (224,224), x) for x in self.images],
                    executed=list(self.executed))


def verify_lock():
    lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())
    for path, expected in lock['files'].items():
        p = Path(path).resolve()
        assert p.is_relative_to(ROOT) and sha(p) == expected, 'LOCK_MISMATCH:'+path
    return lock
