"""Assemble immutable native evaluation prefix without any simulator entry."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
FIT = LINE / 'closed_loop_bench/ordinary_stop_calibration_fit_v2'
INPUT = LINE / 'sft_acceptance/ordinary_stop_row_v12'
DATA = LINE / 'data_pipeline/ordinary_route_teacher_v11/run_001'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent():
    path = FIT / 'reuse.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
    return load('native_fit_source_only', path)


def common():
    text = parent().source('common.py')
    old = "TINY=HERE.parent/'r2r_ce_tiny_v1'"
    assert text.count(old) == 1
    return text.replace(old, "TINY=LINE/'closed_loop_bench/r2r_ce_tiny_v1'")


def worker():
    text = parent().source('evaluate.py')
    marker = '    windows=[c.Window()'
    assert text.count(marker) == 1
    prefix = text[:text.index(marker)]
    assert 'subprocess.Popen' not in prefix and 'build_sim' not in prefix
    suffix = """    replay=c.load('native_full_fit_replay',HERE/'replay.py')
    replay.run(model,policy,forward,c,fingerprint,initial_sha)

if __name__=='__main__':
    main()
"""
    return prefix + suffix
