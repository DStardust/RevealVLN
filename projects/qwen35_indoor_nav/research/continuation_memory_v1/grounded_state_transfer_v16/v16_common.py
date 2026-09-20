"""V16 paths, immutable evidence and explicit source identity."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
DATA_LINE = Path(os.environ.get('V16_ASSET_LINE_ROOT', LINE)).resolve()
BASELINE = '754d574bc98fd49fe30b2be76e30cbc53f0f25c3'
V15 = HERE.parent / 'legal_closed_loop_v15'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

c = load('v16_frozen_io', LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/common.py')
read, write, append, sha = c.read, c.write, c.append, c.sha

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('IMMUTABLE_EVIDENCE_MISMATCH: '+str(path))
    else:
        write(path, value, True)

def source_lock():
    paths = list(HERE.glob('*.py'))
    for folder in ('query_semantics_v11', 'contextual_readout_v10', 'pilot', 'cost_teacher_v13'):
        paths.extend((HERE.parent/folder).glob('*.py'))
    paths.extend((LINE/'sft_acceptance/ordinary_sync_recovery_v1').glob('*.py'))
    paths.extend((V15/'runtime_r2').glob('*.py'))
    paths.extend(V15.glob('*.py'))
    for name in ('compiler.py',):
        paths.append(LINE/'data_pipeline/mechanism_factory_v2'/name)
    for folder in ('data_pipeline/mechanism_runtime_v1','data_pipeline/mechanism_runtime_v1/feedback_generation_v1',
                   'data_pipeline/mechanism_runtime_v1/legal_history_v15','closed_loop_bench/ordinary_cycle_pair_recovery_v5',
                   'closed_loop_bench/r2r_ce_tiny_v1','sft_acceptance/ordinary_baseline_v2'):
        paths.extend((LINE/folder).glob('*.py'))
    paths.append(HERE.parent/'query_reader_repair_v2/model.py')
    return dict(baseline_commit=BASELINE, files={str(p.relative_to(LINE)):sha(p) for p in sorted(set(paths))})

def verify_lock(lock):
    for path, expected in lock['files'].items():
        if sha(LINE/path) != expected:
            raise ValueError('SOURCE_CHANGED: '+path)

def atomic_torch(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as out:
        torch.save(value, out); out.flush(); os.fsync(out.fileno())
    os.link(tmp, path)
    tmp.unlink()
