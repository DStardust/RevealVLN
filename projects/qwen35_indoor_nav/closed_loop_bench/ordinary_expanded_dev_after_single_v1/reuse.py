"""Hash-bound evaluation reuse; force matched single-sample inference only."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'ordinary_expanded_dev_pair_v1/reuse_eval.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == '2f3d2bb62fd25a834773fa59f5ce64e842ddd2d3bd9bfc06712bb8439ce2c85b'
_spec = importlib.util.spec_from_file_location('matched_parent_reuse', PARENT)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)


def source(name):
    result = _parent.source(name)
    if name == 'evaluate.py':
        replacements = {
            '    batch_size=8 if c.parity_accept(checks) else 1':
            "    assert p['parity_pass_batch_size']==p['parity_fallback_batch_size']==1\n    batch_size=1  # Matched comparison; diagnostic parity cannot change this.",
            'dict(passed=batch_size==8,selected_batch_size=batch_size,checks=checks,':
            'dict(passed=c.parity_accept(checks),selected_batch_size=batch_size,checks=checks,comparison_batch_size_forced=1,',
        }
        for old, new in replacements.items():
            assert result.count(old) == 1, 'SOURCE_ANCHOR_CHANGED'
            result = result.replace(old, new)
    return result


def execute(name, namespace):
    exec(compile(source(name), str(Path(namespace['__file__'])) + ':parent', 'exec'), namespace)
