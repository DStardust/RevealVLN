"""Record main-agent phase-specific acceptance after CPU test delivery."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import sha, ROOT, LINE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, cpu = args.output.resolve(), args.cpu_result.resolve()
    assert out.is_relative_to(HERE) and cpu.is_relative_to(HERE)
    verdict = json.loads(cpu.read_text())
    assert verdict['pass']
    tested = json.loads((cpu.parent/'CODE_LOCK.json').read_text())
    assert all(sha(HERE/name) == value for name, value in tested.items()), 'CODE_CHANGED_AFTER_CPU_TEST'
    config = json.loads((out/'CONFIG_DRAFT.json').read_text())
    assert config['phase'] == 'existing_family_backend_smoke' and not config['runtime_allowed']
    config['runtime_allowed'] = True
    with (out/'EXECUTION_CONFIG.json').open('x') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    binary = (LINE/'.envs/q35n_habitat_v017_g0r/bin/python3').resolve()
    assert binary.is_relative_to(ROOT)
    auth = {'approved': True, 'scope': 'nine_existing_family_traces_then_V4_readback_only',
            'config_sha256': sha(out/'EXECUTION_CONFIG.json'),
            'cpu_result_sha256': sha(cpu), 'code_sha256': {p.name: sha(p) for p in HERE.glob('*.py')},
            'python_realpath': str(binary), 'python_sha256': sha(binary),
            'environment_basis': 'previously_accepted_G0R_independent_environment_unchanged_no_install',
            'gpu_lease_recheck_by_supervisor_required': True, 'P0_authorized': False,
            'scientific_pass': False}
    with (out/'EXECUTION_AUTH.json').open('x') as f:
        json.dump(auth, f, indent=2)
    print(json.dumps({'approved': True, 'phase': config['phase'], 'gpu_device': config['gpu_device']}))


if __name__ == '__main__':
    main()
