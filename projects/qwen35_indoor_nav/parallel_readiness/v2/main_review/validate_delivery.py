"""Independent CPU review; never executes branch artifact-writing entrypoints."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def verify(directory):
    entries = []
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split(None, 1)
        path = (directory / relative).resolve()
        assert path.is_relative_to(directory.resolve()), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        entries.append(relative)
    return {'pass': True, 'files': len(entries),
            'manifest_sha256': hashlib.sha256((directory / 'SHA256SUMS').read_bytes()).hexdigest()}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    names = ('shortcut_audit', 'multifamily_plan', 'loader')
    before = {name: verify(ROOT / name) for name in names}
    audit = module('review_audit', ROOT / 'shortcut_audit/audit.py')
    planner = module('review_planner_tests', ROOT / 'multifamily_plan/test_dry_run.py')
    module('loader', ROOT / 'loader/loader.py')
    loader_tests = module('review_loader_tests', ROOT / 'loader/test_loader.py')
    test_results = {}
    for name, m in zip(names, (audit, planner, loader_tests)):
        log = io.StringIO()
        r = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
        test_results[name] = {'tests': r.testsRun, 'failures': len(r.failures),
                              'errors': len(r.errors), 'pass': r.wasSuccessful()}
        (HERE / (name + '_TEST_LOG.txt')).write_text(log.getvalue())
        assert r.testsRun > 0 and r.wasSuccessful(), name
    after = {name: verify(ROOT / name) for name in names}
    assert before == after
    source = verify(ROOT.parents[1] / 'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1')
    result = {'decision': 'INDEPENDENT_CPU_DELIVERY_CHECK_PASS', 'tests': test_results,
              'branch_hashes': after, 'branch_hashes_unchanged': True,
              'accepted_source': source, 'new_simulator_runs': 0, 'new_model_runs': 0,
              'sft_results_read': False, 'scientific_pass': False}
    with (HERE / 'VALIDATION.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
