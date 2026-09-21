"""Paths and immutable outputs for the new, CPU-prepared policy study."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
V16 = HERE.parent / 'grounded_state_transfer_v16'
SOURCE = V16 / 'b2_matched_control_v1/runs/matched_001'
sys.path.insert(0, str(V16))
from v16_common import load, read, write, sha, digest, immutable, atomic_torch, c

old = load('evidence_source_objective', V16 / 'objective.py')
checker = load('evidence_frozen_checker', V16 / 'evaluator_v16.py')
# Frozen loaders prepend their own directories; this version owns CLI imports.
sys.path.insert(0, str(HERE))


def verify_binding(run):
    binding = read(run / 'BINDING.json')
    for path, expected in binding['files'].items():
        if sha(Path(path)) != expected:
            raise ValueError('BOUND_FILE_CHANGED:' + path)
    return binding
