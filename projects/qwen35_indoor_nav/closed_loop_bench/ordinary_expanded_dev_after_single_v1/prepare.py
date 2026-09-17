"""Freeze a fresh matched case; no training, simulator, or GPU initialization."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
AFTER = HERE.parent / 'ordinary_expanded_dev_after_v1'
BEFORE = HERE.parent / 'ordinary_expanded_dev_before_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    with Path(path).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    assert not (HERE / 'PROTOCOL.json').exists(), 'ALREADY_FROZEN'
    s = importlib.util.spec_from_file_location('single_reuse_test', HERE / 'reuse.py')
    reuse = importlib.util.module_from_spec(s); s.loader.exec_module(reuse)
    checks = []
    for name in reuse._parent.HASHES:
        text = reuse.source(name); ast.parse(text)
        if name != 'evaluate.py':
            assert text == reuse._parent.source(name)
        checks.append(name + ':syntax_and_scoped_reuse')
    # Execute only the two selection/receipt expressions, never load torch here.
    tree = ast.parse(reuse.source('evaluate.py'))
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assignments = [n for n in ast.walk(main_node) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'batch_size' for t in n.targets)]
    assert len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant) and assignments[0].value.value == 1
    assert 'passed=c.parity_accept(checks)' in reuse.source('evaluate.py')
    checks += ['forced_batch_one_no_result_dependent_switch', 'parity_pass_report_independent_of_batch_choice']
    before = read(BEFORE / 'run_001/RESULT.json')
    after = read(AFTER / 'run_001/RESULT.json')
    assert before['completed'] == after['completed'] == 100
    assert before['trace_audit_passed'] and after['trace_audit_passed']
    assert before['selected_batch_size'] == 1 and after['selected_batch_size'] == 8
    checks.append('existing_comparison_mismatch_verified')
    p = read(AFTER / 'PROTOCOL.json')
    assert p['checkpoint_sha256'] == 'c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    assert sha(p['checkpoint']) == p['checkpoint_sha256'] and p['seed'] == 1209
    p.update(id='Q35N_EXPANDED_DEV_AFTER_SINGLE_V1', parity_pass_batch_size=1,
             comparison_batch_size_forced=1, checkpoint_role='after_matched_single',
             main_criterion='Correct matched batch-1 development comparison; not scientific PASS')
    save(HERE / 'PROTOCOL.json', p)
    for name in ('EPISODES_PRIVILEGED.json', 'GEOMETRY_PREFLIGHT.json', 'PARITY_FIXTURES.json'):
        # Identical serialized bytes, not a new selection.
        with (HERE / name).open('xb') as f: f.write((AFTER / name).read_bytes())
        assert sha(HERE / name) == sha(AFTER / name)
    assert sha(HERE / 'EPISODES_PRIVILEGED.json') == sha(BEFORE / 'EPISODES_PRIVILEGED.json') == '2f41a84a575aabe3ddbb5d9d41e29bd9c779eff93b673b80d4c06db0ba953b16'
    checks += ['same_checkpoint_and_seed', 'identical_hundred_episode_input_and_geometry']
    files = dict(read(AFTER / 'SOURCE_LOCK.json')['files'])
    for path, digest in files.items():
        assert Path(path).resolve(strict=True).is_relative_to(ROOT) and sha(path) == digest, path
    checks.append('all_inherited_frozen_hashes_unchanged')
    save(HERE / 'CPU_TEST_RESULT.json', dict(passed=True, checks=checks, count=len(checks), optimizer_updates=0))
    for path in list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + [HERE / 'SPEC_ZH.md']:
        files[str(path)] = sha(path)
    save(HERE / 'SOURCE_LOCK.json', dict(files=files, scope='Only forced batch-one comparison correction; all historical sources frozen', training_updates=0))
    save(HERE / 'MAIN_REVIEW.json', dict(status='PASS_FOR_BOUNDED_CORRECTNESS_RETEST', user_request='检查并做出针对性修改',
         source_lock_sha256=sha(HERE / 'SOURCE_LOCK.json'), input_sha256=sha(HERE / 'EPISODES_PRIVILEGED.json'),
         cpu_test_count=len(checks), automatic_training=False, automatic_retry=False, new_novelty_claim=False))
    print(json.dumps(read(HERE / 'CPU_TEST_RESULT.json'), ensure_ascii=False))


if __name__ == '__main__': main()
