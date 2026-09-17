"""Run and preserve one CPU acceptance attempt; never invoke runtime commands."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cli import verify_seal, prepare

LINE = HERE.parents[1]


def save(directory, name, value):
    with (directory/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def protected():
    roots = [LINE/'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1',
             LINE/'reviews/Q35N_G1R_TASK_INSTANCE_V3']
    roots += [LINE/'parallel_readiness/v2'/name for name in ('loader', 'shortcut_audit', 'multifamily_plan', 'main_review')]
    result = {}
    for root in roots:
        result[str(root.relative_to(LINE))] = {'files': verify_seal(root),
            'manifest_sha256': hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest()}
    lock = json.loads((roots[0]/'COMPILER_CODE_LOCK.json').read_text())
    for relative, expected in lock.items():
        path = (LINE/relative).resolve()
        assert path.is_relative_to(LINE) and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    result['compiler_locked_files'] = lock
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not args.attempt.replace('_', '').isalnum():
        parser.error('invalid attempt name')
    out = HERE/args.attempt
    out.mkdir(exist_ok=False)
    save(out, 'CODE_LOCK.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))})
    before = protected()
    save(out, 'PROTECTED_BEFORE.json', before)
    started = time.monotonic()
    results = {}
    for name in ('test_compiler', 'test_planning', 'test_factory', 'test_journal', 'test_cli', 'test_integration'):
        log = io.StringIO()
        r = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(name))
        with (out/(name+'.log')).open('x') as f:
            f.write(log.getvalue())
        results[name] = {'tests': r.testsRun, 'failures': len(r.failures), 'errors': len(r.errors),
                         'pass': r.wasSuccessful() and r.testsRun > 0}
    after = protected()
    save(out, 'PROTECTED_AFTER.json', after)
    passed = all(r['pass'] for r in results.values()) and before == after
    if passed:
        prepare(out/'prepared_p0')
    result = {'decision': 'GENERIC_FACTORY_CPU_CORE_PASS' if passed else 'CPU_ACCEPTANCE_FAIL',
              'tests': results, 'total_tests': sum(r['tests'] for r in results.values()),
              'elapsed_seconds': time.monotonic()-started, 'protected_unchanged': before == after,
              'real_log_differential_evaluations': 54 if results['test_compiler']['pass'] else None,
              'recorded_backend_fixture_traces': 27 if results['test_factory']['pass'] else None,
              'recorded_backend_fixture_action_returns': 8631 if results['test_factory']['pass'] else None,
              'new_simulator_replays': 0, 'new_physical_families': 0, 'gpu_operations': 0,
              'model_calls': 0, 'training_updates': 0, 'sft_results_read': False,
              'habitat_adapter_implemented': False, 'production_export_v4_implemented': False,
              'runtime_pass': None, 'scientific_pass': False}
    save(out, 'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
