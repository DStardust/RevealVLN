import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WF = HERE.parent
RUNTIME = WF.parent
LINE = RUNTIME.parents[1]
ROOT = LINE.parents[1]
OUT = HERE / 'run_v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def verify_inputs():
    lock = read(OUT / 'INPUT_LOCK.json')
    assert 1 <= len(lock) <= 4096
    for p, expected in lock.items():
        path = Path(p)
        assert path == path.resolve() and path.is_relative_to(ROOT)
        assert sha(path) == expected, str(path)
    approval = read(HERE / 'MAIN_AGENT_APPROVAL.json')
    assert approval == dict(approved=True, input_lock_sha256=sha(OUT / 'INPUT_LOCK.json'),
                            node='Q35N_SPECIAL_DIVERSITY_V1', gpu=1,
                            authorization='user_20260911_replace_opencode_diversity_and_teacher_report')
    cfg = read(OUT / 'EXECUTION_CONFIG.json')
    assert cfg['runtime_allowed'] and not cfg['training_allowed']
    assert cfg['gpu_device'] == 1 and cfg['supervision_wall_seconds'] == 3900
    return cfg
