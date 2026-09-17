"""No-overwrite CPU preflight and main-agent admission for exactly one P0 batch."""
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
sys.path[:0] = [str(RUNTIME/'p0_driver_v1'), str(RUNTIME)]
from prepare import sha, sealed, ROOT, LINE
import supervisor


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    sealed(RUNTIME)
    sealed(LINE/'data_pipeline/mechanism_factory_v2')
    sealed(LINE/'parallel_readiness/v2/multifamily_plan')
    assert not (HERE/'run').exists(), 'NO_OVERWRITE_OR_RETRY'
    before = {str(p.relative_to(LINE)): sha(p/'SHA256SUMS') for p in
              [RUNTIME, LINE/'data_pipeline/mechanism_factory_v2',
               LINE/'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1', LINE/'parallel_readiness/v2/multifamily_plan']}
    save(HERE/'PROTECTED_BEFORE.json', before)
    results = {}
    for name in ('test_prepare', 'test_family_job', 'test_p0_supervisor'):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(name))
        with (HERE/(name+'.log')).open('x') as f:
            f.write(stream.getvalue())
        results[name] = {'tests': result.testsRun, 'pass': result.wasSuccessful() and result.testsRun > 0}
    save(HERE/'CPU_RECHECK.json', results)
    assert sum(r['tests'] for r in results.values()) == 32 and all(r['pass'] for r in results.values())
    cfg = json.loads((RUNTIME/'p0_batch_v1/CONFIG_DRAFT.json').read_text())
    source = (ROOT/'data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz').resolve()
    assert source.is_relative_to(ROOT) and sha(source) == cfg['source_sha256']
    for row in cfg['candidates']:
        for path, value in row['assets'].items():
            p = Path(path).resolve()
            assert p.is_relative_to(ROOT) and str(p) == path and sha(p) == value
    cfg.update(runtime_allowed=True, smoke_result_sha256=sha(RUNTIME/'smoke_v1/result.json'))
    out = HERE/'run'
    out.mkdir()
    save(out/'EXECUTION_CONFIG.json', cfg)
    files = list(RUNTIME.glob('*.py'))+list((RUNTIME/'p0_driver_v1').glob('*.py'))
    binary = (LINE/'.envs/q35n_habitat_v017_g0r/bin/python3').resolve()
    auth = {'approved': True, 'P0_authorized': True, 'P1_authorized': False,
            'config_sha256': sha(out/'EXECUTION_CONFIG.json'),
            'code_sha256': {str(p.relative_to(RUNTIME)): sha(p) for p in files},
            'python_realpath': str(binary), 'python_sha256': sha(binary),
            'CPU_recheck_sha256': sha(HERE/'CPU_RECHECK.json'),
            'node_authorization_sha256': sha(LINE/'authorizations/MECHANISM_P0_EXECUTION_V1.json'),
            'admission_script_sha256': sha(Path(__file__)),
            'training_allowed': False, 'scientific_pass': False}
    supervisor.verify_admission(out, cfg, auth)
    state = supervisor.gpu_state(2)
    assert state['uuid'] == 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
    assert state['utilization'] == 0 and state['memory_mib'] < 1024
    pids = {int(p['pid']) for p in state['processes']}
    assert pids <= {2781015, 2900192}
    identities = {}
    for pid in pids:
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart=,args='], text=True, timeout=10)
        assert 'eval.scripts.evaluate_pointgoal' in command and '--device cuda:0' in command
        identities[str(pid)] = command.strip()
    save(HERE/'GPU_ADMISSION_REVIEW.json', {'state': state, 'identities': identities, 'no_process_stopped': True})
    save(out/'EXECUTION_AUTH.json', auth)
    print(json.dumps({'approved': True, 'out': str(out), 'candidate_bundles': 5,
                      'cpu_tests': 32, 'budget_changed': False}))


if __name__ == '__main__':
    main()
