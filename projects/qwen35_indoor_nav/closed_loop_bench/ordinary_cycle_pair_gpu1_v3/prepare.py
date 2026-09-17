"""A resource-only migration of the sealed shared-process comparison."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ordinary_cycle_pair_v2'


def write(name, value):
    with (HERE / name).open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def main():
    original = {}
    for name in ('common.py', 'evaluate.py', 'executor.py', 'aggregate.py', 'launch.py',
                 'path_metrics.py', 'official_distance.py', 'cycle_policy.py', 'review.py'):
        text = (OLD / name).read_text()
        original[name] = hashlib.sha256(text.encode()).hexdigest()
        if name == 'launch.py':
            assert text.count("HERE/'gpu4_eval.lock'") == 1
            text = text.replace("HERE/'gpu4_eval.lock'", "HERE/'gpu1_eval.lock'")
        ast.parse(text)
        write(name, text)
    p = json.loads((OLD / 'PROTOCOL.json').read_text())
    p.update(id='Q35N_ORDINARY_CYCLE_PAIR_GPU1_V3', gpu=1,
             gpu_uuid='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8', wall_seconds=4100)
    write('PROTOCOL.json', p)
    for name in ('EPISODES_PRIVILEGED.json', 'GEOMETRY_PREFLIGHT.json', 'PARITY_FIXTURES.json'):
        write(name, (OLD / name).read_text())
    c = runpy.run_path(str(HERE / 'common.py'))
    lock = json.loads((OLD / 'SOURCE_LOCK.json').read_text())
    for path in [*HERE.glob('*.py'), *HERE.glob('*.json'), HERE / 'SPEC_ZH.md']:
        lock['files'][str(path.resolve())] = c['sha'](path)
    lock['migration_source_sha256'] = original
    write('SOURCE_LOCK.json', lock)
    result = json.loads((OLD / 'CPU_TEST_RESULT.json').read_text())
    result.update(resource_only_migration=True, paired_algorithm_and_assets_unchanged=True)
    write('CPU_TEST_RESULT.json', result)
    print('Prepared identical shared-process pair on GPU1')


if __name__ == '__main__':
    main()
