"""Local paths and atomic records for the StreamVLN transfer experiment."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REPO = HERE.parents[4]
ASSETS = Path('/mnt/data_nas/deeprobotics/daiyang/vla')
CODE = ASSETS / 'third_party/B33_STREAMVLN_V1'
MODEL = ASSETS / 'models/B33_STREAMVLN_V1'
PYTHON = ASSETS / '.envs/b33_streamvln_v1/bin/python'
CONTROL_PYTHON = ASSETS / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
ORIGIN = ASSETS / 'artifacts/experiments/B33_STREAMVLN_BASELINE_ADMISSION_V1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(run):
    for path, expected in read(run / 'SOURCE_LOCK.json')['files'].items():
        if sha(path) != expected:
            raise ValueError('SOURCE_CHANGED: ' + path)


def setup_imports(run):
    import sys
    sys.path[:0] = [str(CODE / 'vendor_simulator'), str(CODE / 'vendor_depth'), str(CODE / 'streamvln'), str(CODE)]
    for key in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'CUDA_CACHE_PATH'):
        os.environ[key] = str(run / 'runtime_cache' / key.lower())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
