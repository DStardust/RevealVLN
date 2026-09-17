"""Frozen, separately named CPU acceptance attempts before renderer admission."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import sealed, sha, LINE, DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    name = parser.parse_args().attempt
    assert name.replace('_', '').isalnum()
    out = HERE/name
    out.mkdir(exist_ok=False)
    roots = [DATA, LINE/'data_pipeline/mechanism_factory_v2', LINE/'reviews/Q35N_G1R_TASK_INSTANCE_V3']
    for root in roots:
        sealed(root)
    before = {str(root.relative_to(LINE)): sha(root/'SHA256SUMS') for root in roots}
    with (out/'CODE_LOCK.json').open('x') as f:
        json.dump({p.name: sha(p) for p in sorted(HERE.glob('*.py'))}, f, indent=2)
    results = {}
    for module in ('test_backend', 'test_guard', 'test_export', 'test_supervisor'):
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(module))
        with (out/(module+'.log')).open('x') as f:
            f.write(log.getvalue())
        results[module] = {'tests': result.testsRun, 'failures': len(result.failures),
                           'errors': len(result.errors), 'pass': result.wasSuccessful() and result.testsRun > 0}
    for root in roots:
        sealed(root)
    after = {str(root.relative_to(LINE)): sha(root/'SHA256SUMS') for root in roots}
    passed = before == after and all(r['pass'] for r in results.values())
    data = {'decision': 'CPU_INTEGRATION_PASS' if passed else 'CPU_INTEGRATION_FAIL', 'pass': passed,
            'tests': results, 'total_tests': sum(r['tests'] for r in results.values()),
            'protected_before': before, 'protected_after': after,
            'gpu_operations': 0, 'simulator_runs': 0, 'sft_results_read': False, 'scientific_pass': False}
    with (out/'result.json').open('x') as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
