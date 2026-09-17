import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
DISCOVERY = OUT.parent/'Q35N_G1R_TASK_INSTANCE_V3'


def main():
    assert (DISCOVERY/'FROZEN_CANDIDATE.json').is_file()
    assert json.loads((DISCOVERY/'LEASE_RESTORED.json').read_text())['restored']
    for name, expected in json.loads((DISCOVERY/'CODE_LOCK.json').read_text()).items():
        assert hashlib.sha256((DISCOVERY/name).read_bytes()).hexdigest() == expected
    tests = subprocess.run([sys.executable, '-I', '-S', str(LINE/'data_pipeline/mechanism_family_v1/test_compiler.py')], capture_output=True, text=True)
    assert tests.returncode == 0, tests.stderr
    with (OUT/'UNIT_TESTS.json').open('x') as f: json.dump({'returncode': tests.returncode, 'output': tests.stdout+tests.stderr, 'synthetic_unit_tests_only': True}, f, indent=2)
    files = list((LINE/'data_pipeline/mechanism_family_v1').glob('*.py'))+list((LINE/'data_pipeline/mechanism_family_v1').glob('*.md'))
    with (OUT/'COMPILER_CODE_LOCK.json').open('x') as f:
        json.dump({str(p.relative_to(LINE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, f, indent=2)
    s = importlib.util.spec_from_file_location('bounded_runner', OUT.parent/'Q35N_G1R_COVERAGE_V1/run.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.OUT = OUT; m.main()


if __name__ == '__main__': main()
