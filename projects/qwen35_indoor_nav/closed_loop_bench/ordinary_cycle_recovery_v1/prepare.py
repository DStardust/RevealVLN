"""Materialize the accepted evaluation code with one declared policy intervention."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
BASE = HERE.parent / 'ordinary_expanded_dev_after_single_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (HERE / name).open('x') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    assert not (HERE / 'run_001').exists()
    source = runpy.run_path(str(BASE / 'reuse.py'))['source']
    source_hashes = {}
    for name in ('common.py', 'evaluate.py', 'executor.py', 'aggregate.py',
                 'launch.py', 'path_metrics.py', 'official_distance.py'):
        text = source(name)
        source_hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        if name == 'evaluate.py':
            text = replace(text, "    windows=[c.Window() for _ in range(p['lanes'])]",
                "    cycle=c.load('cycle_policy',HERE/'cycle_policy.py')\n"
                "    guards=[cycle.CycleRecovery() for _ in range(p['lanes'])]\n"
                "    windows=[c.Window() for _ in range(p['lanes'])]")
            text = replace(text, '        episode_steps[lane]=0',
                '        guards[lane].reset()\n        episode_steps[lane]=0')
            text = replace(text, '                    action=c.ACTIONS[max(range(4),key=values.__getitem__)]',
                '                    window=windows[lane];tick_cycle=time.perf_counter()\n'
                '                    action,cycle_info=guards[lane].choose(window.instruction,window.images,window.executed,values)\n'
                '                    cycle_info["cycle_seconds"]=time.perf_counter()-tick_cycle')
            text = replace(text, 'executed_history=len(windows[lane].executed)))',
                'executed_history=len(windows[lane].executed),**cycle_info))')
        elif name == 'launch.py':
            text = replace(text, "OUT=HERE/'run_001'", "OUT=HERE/'run_001'\n"
                "tree=c.load('tree_size',HERE.parent/'ordinary_stop_calibration_fit_v2/tree_size.py')")
            text = replace(text, "    status=json.loads((c.TRAIN/'formal/STATUS.json').read_text())\n"
                "    return dict(status=status,processes=[identity(pid) for pid in (1151310,1151311,1151312,1151313)])",
                "    return dict(status='No external training process supervision in this run',processes=[])")
            text = replace(text, "HERE/'gpu1_eval.lock'", "HERE/'gpu4_eval.lock'")
            text = replace(text, '# Training may not yet have started or already naturally stopped; GPU1 independent.',
                '# Use only the empty device; external process identities are recorded in GPU snapshots.')
            text = replace(text, "            size=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())",
                '            size,transient_misses=tree.tree_size(OUT)')
            text = replace(text, 'own_pids=sorted(pids),misplaced=misplaced,foreign=foreign,stop_reason=reason)',
                'own_pids=sorted(pids),misplaced=misplaced,foreign=foreign,stop_reason=reason,transient_stat_misses=transient_misses)')
            start = text.index('        unchanged=all(')
            end = text.index('        resource_reasons=', start)
            text = text[:start] + '        unchanged=live_unchanged=None  # No claim about external process lifetimes.\n' + text[end:]
        ast.parse(text)
        write(name, text)
    p = json.loads((BASE / 'PROTOCOL.json').read_text())
    original = p.copy()
    p.update(id='Q35N_ORDINARY_CYCLE_RECOVERY_V1', gpu=4,
             gpu_uuid='GPU-e458147b-4739-d22a-e764-50743cff4a11', wall_seconds=2700,
             greedy=False, native_policy_greedy=True, checkpoint_role='best4k_cycle_recovery',
             checkpoint_selection='Same frozen best4k as historical matched development baseline',
             main_criterion='SR > .21, SPL >= .18016983923442284, nDTW >= .3558797007353029; matched prefixes required',
             recovery='Least attempted motion on repeated complete causal input; logits break ties; native STOP preserved')
    write('PROTOCOL.json', p)
    for name in ('EPISODES_PRIVILEGED.json', 'PARITY_FIXTURES.json', 'GEOMETRY_PREFLIGHT.json'):
        write(name, (BASE / name).read_text())
    # Keep inherited code/model integrity records, but do not access unused test
    # datasets, unrelated scene assets or checkpoints in this train-only node.
    inherited = json.loads((BASE / 'SOURCE_LOCK.json').read_text())['files']
    files = {}
    for path, digest in inherited.items():
        f = Path(path)
        if '/val_unseen/' in path:
            continue
        if '/scene_datasets/mp3d/' in path and f.parent.name not in p['houses']:
            continue
        if f.suffix == '.pt' and path != p['checkpoint']:
            continue
        files[path] = digest
    files[p['checkpoint']] = p['checkpoint_sha256']
    extras = [BASE / 'SOURCE_LOCK.json', BASE / 'PROTOCOL.json', BASE / 'reuse.py',
              HERE.parent / 'ordinary_expanded_dev_pair_v1/reuse_eval.py',
              HERE.parent / 'ordinary_stop_calibration_fit_v2/tree_size.py', Path(p['gt_path'])]
    fixtures = json.loads((HERE / 'PARITY_FIXTURES.json').read_text())
    extras.extend(Path(path) for group in fixtures for row in group for path in row['images'])
    extras.extend(HERE.glob('*.py'))
    extras.extend(HERE / name for name in ('SPEC_ZH.md', 'PROTOCOL.json', 'EPISODES_PRIVILEGED.json',
                                         'PARITY_FIXTURES.json', 'GEOMETRY_PREFLIGHT.json'))
    for path in extras:
        files[str(path.resolve())] = sha(path)
    write('SOURCE_LOCK.json', dict(files=files, scope='Inherited runtime integrity; unused evaluation data/assets/checkpoints excluded',
                                   baseline_expanded_source_sha256=source_hashes))
    test = subprocess.run([sys.executable, '-I', '-S', '-B', str(HERE / 'test_cycle_policy.py')],
                          capture_output=True, text=True, timeout=60)
    assert test.returncode == 0, test.stdout + test.stderr
    write('CPU_TEST_RESULT.json', dict(passed=True, actual_trace_and_contract_tests=3,
        output=test.stdout + test.stderr, protocol_changes={key: [original.get(key), value]
        for key, value in p.items() if original.get(key) != value},
        generated_sources_parse=True, source_lock_count=len(files),
        geometry_preflight_reused='Identical episodes and simulator geometry; real resets recheck distances'))
    print(json.dumps(dict(prepared=True, files=len(files), cpu_tests=3)))


if __name__ == '__main__':
    main()
