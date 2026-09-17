"""CPU provenance/transport tests and a single immutable diagnostic seal."""
import ast
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (HERE / 'SOURCE_LOCK.json').exists() and not (HERE / 'CPU_TEST_RESULT.json').exists()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    source = load('native_assembly_prepare', HERE / 'source.py')
    replay = load('native_replay_prepare', HERE / 'replay.py')
    common = load('native_common_prepare', HERE / 'common.py')
    launch = load('native_launch_prepare', HERE / 'launch.py')
    audit = load('native_audit_prepare', HERE / 'audit.py')
    assert common.TINY == LINE / 'closed_loop_bench/r2r_ce_tiny_v1'
    original = source.parent().source('evaluate.py')
    cut = original.index('    windows=[c.Window()')
    assert source.worker().startswith(original[:cut])
    assert 'subprocess.Popen' not in source.worker() and 'build_sim' not in source.worker()
    assert 'batch_size=1  # Matched comparison;' in source.worker()
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    ast.parse(source.worker())
    ast.parse(source.common())
    inputs = rows(source.INPUT / 'POLICY_INPUTS.jsonl')
    labels = rows(source.INPUT / 'SUPERVISION_ONLY.jsonl')
    assert len(inputs) == len(labels) == 5487
    assert [x['record_id'] for x in inputs] == [x['record_id'] for x in labels]
    assert len({x['record_id'] for x in inputs}) == 5487
    contracts = load('frozen_payload_contract', source.DATA.parent / 'contracts.py')
    episodes = read(source.FIT / 'EPISODES_PRIVILEGED.json')
    assert len(episodes) == 64
    originals = {}
    for lane in sorted((source.FIT / 'run_001/lanes').glob('lane_*')):
        initial = {r['index']: r['rgb_sha256'] for r in rows(lane / 'INTERFACE.jsonl')}
        policies = collections.defaultdict(list)
        steps = collections.defaultdict(list)
        for row in rows(lane / 'POLICY_STEPS.jsonl'):
            policies[row['index']].append(row)
        for row in rows(lane / 'STEPS_PRIVILEGED.jsonl'):
            steps[row['index']].append(row)
        for index, policy_rows in policies.items():
            seen = [initial[index]]
            executed = []
            assert len(policy_rows) == len(steps[index])
            for t, (p, s) in enumerate(zip(policy_rows, steps[index])):
                assert p['step'] == s['step'] == t + 1 and p['action'] == s['action']
                payload = contracts.payload(episodes[index]['instruction']['instruction_text'], seen[-2:], executed[-8:])
                originals[index, t] = (payload, p['logits'])
                seen.append(s['rgb_sha256'])
                executed.append(p['action'])
    assert len(originals) == 7225
    count = 0
    for row, label in zip(inputs, labels):
        replay.validate_input(row)
        payload = {k: v for k, v in row.items() if k != 'record_id'}
        assert contracts.key(payload) == row['record_id']
        for occurrence in label['occurrences']:
            observed, logits = originals[occurrence['episode'], occurrence['step']]
            assert payload == observed and occurrence['original_logits'] == logits
            assert logits == label['occurrences'][0]['original_logits']
            count += 1
    assert count == 7225
    houses = {r['house'] for r in labels}
    assert len(houses) == 16
    for name in ('ordinary_expanded_dev_after_single_v1', 'r2r_ce_full_v2'):
        assert not houses & set(read(LINE / 'closed_loop_bench' / name / 'PROTOCOL.json')['houses'])
    digests = sorted({x for row in inputs for x in row['rgb_sha256']})
    for digest in digests:
        path = source.DATA / 'content' / (digest + '.png')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            assert rgb.size == (224, 224) and hashlib.sha256(rgb.tobytes()).hexdigest() == digest
    rejects = 0
    for field in ('target', 'distance', 'house', 'future_frame', 'original_logits', 'position'):
        bad = dict(inputs[0], **{field: 'FORBIDDEN'})
        try:
            replay.validate_input(bad)
        except AssertionError:
            rejects += 1
    assert rejects == 6
    assert audit.compare([1., 2., 3., 4.], [1., 2., 3., 4.], 1e-5) == (0., True, True)
    assert not audit.compare([1., 2., 3., 4.], [1., 2., 3., 4.001], 1e-5)[2]
    assert not audit.compare([0., 0., -1., -2.], [0., 1e-6, -1., -2.], 1e-5)[2]
    try:
        audit.compare([1., 2., 3., 4.], [1., 2., float('nan'), 4.], 1e-5)
        raise RuntimeError('NONFINITE_ACCEPTED')
    except AssertionError:
        pass
    model = load('cpu_native_model', LINE / 'sft_acceptance/ordinary_sync_recovery_v1/model.py')
    processor = AutoProcessor.from_pretrained(model.MODEL, local_files_only=True, trust_remote_code=False)
    ids = read(LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json')['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens': list(ids)})
    assert all(processor.tokenizer.convert_tokens_to_ids(k) == v for k, v in ids.items())
    collate = model.make_collate(processor.tokenizer.pad_token_id,
        {a: ids[model.EXEC_TOKENS[a]] for a in model.ACTIONS[:-1]}, ids['<NAV_ACTION_QUERY>'],
        read(model.MODEL / 'config.json')['image_token_id'])
    chosen = list(range(16)) + [300 * i for i in range(1, 17)]
    class Store:
        def __init__(self, native): self.native = native
        def get(self, i, t):
            row = inputs[chosen[i]]
            images = []
            for digest in row['rgb_sha256']:
                with Image.open(source.DATA / 'content' / (digest + '.png')) as image:
                    images.append(image.convert('RGB'))
            if not self.native:
                return dict(instruction=row['instruction'], images=images, executed=row['executed_actions'])
            window = common.Window()
            window.instruction = row['instruction']
            window.images = [im.tobytes() for im in images]
            window.executed = row['executed_actions']
            return window.item()
    samples = [dict(record_idx=i, t=0, target=0, weight=1.) for i in range(32)]
    a = model.DecisionDataset(samples, Store(False), processor)
    b = model.DecisionDataset(samples, Store(True), processor)
    for i in range(32):
        left, right = collate([a[i]]), collate([b[i]])
        assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
    tests = runpy.run_path(str(LINE / 'sft_acceptance/ordinary_prefix_history8_v1/tests_r1.py'), run_name='READ_ONLY_CPU_FUNCTIONS')
    process_checks = tests['process_tests']()
    assert len(process_checks) == 7
    budget = dict(id='Q35N_FULL_FIT_NATIVE_REPLAY_V1', runtime_allowed=True, wall_seconds=1200,
        worker_wall_seconds=1100, gpu_gib=28, model_gib=25, cpu_rss_gib=64, output_gib=1,
        forward_decisions=5519, replay_inputs=5487, fixture_forward_decisions=32,
        optimizer_updates=0, backward_calls=0, simulator_actions=0, max_abs_logit_error=1e-5,
        all_argmax_same_required=True, checkpoint_sha256=read(source.FIT / 'PROTOCOL.json')['checkpoint_sha256'],
        no_automatic_training=True, no_novelty_claim=True)
    good = dict(gpus={launch.GPU: dict(memory_mib=1000, pids=[100])}, wall_seconds=1,
                rss_bytes=1000, output_bytes=1000, forward_decisions=1)
    assert launch.violation(good, [100], budget) is None
    cases = [('wall_seconds', 1101, 'WALL_BUDGET'), ('rss_bytes', 65 * 1024**3, 'CPU_RSS_BUDGET'),
             ('output_bytes', 2 * 1024**3, 'OUTPUT_BUDGET'), ('forward_decisions', 5520, 'FORWARD_BUDGET')]
    for key, value, expected in cases:
        assert launch.violation(dict(good, **{key: value}), [100], budget) == expected
    assert launch.violation(good, [], budget) == 'FOREIGN_GPU1_CONTEXT'
    assert launch.violation(dict(good, gpus={launch.GPU: dict(memory_mib=29000, pids=[100])}), [100], budget) == 'GPU_MEMORY_BUDGET'
    assert launch.violation(dict(good, gpus={**good['gpus'], 'ANOTHER_GPU': dict(memory_mib=1, pids=[100])}), [100], budget) == 'OWN_GPU_MAPPING_MISMATCH'
    assert not torch.cuda.is_initialized()
    files = dict(read(source.INPUT / 'SOURCE_LOCK.json')['files'])
    for path, digest in files.items():
        assert sha(Path(path)) == digest, path
    for name in ('PROTOCOL.json', 'PARITY_FIXTURES.json'):
        with (HERE / name).open('xb') as stream:
            stream.write((source.FIT / name).read_bytes())
    (HERE / 'run_001').mkdir()
    (LINE / '.tfn1').mkdir(exist_ok=True)
    for relative in ('hf', 'xdg/torch/kernels', 'torch', 'cuda', 'inductor', 'triton'):
        (HERE / 'cache' / relative).mkdir(parents=True, exist_ok=True)
    caches = []
    for path in sorted((source.FIT / 'run_001/cache/triton').rglob('*.autotune.json')):
        target = HERE / 'cache/triton' / path.relative_to(source.FIT / 'run_001/cache/triton')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(path.read_bytes())
        caches.append(dict(source=str(path), target=str(target), sha256=sha(path)))
        files[str(path)] = sha(path)
    assert len(caches) == 7
    write(HERE / 'CACHE_PROVENANCE.json', dict(files=caches, source_read_only=True))
    write(HERE / 'REPLAY_PROTOCOL.json', budget)
    write(HERE / 'CPU_TEST_RESULT.json', dict(status='PASS', unix=time.time(),
        input_rows=5487, original_occurrences=7225, fit_houses=16, decoded_rgb_files=len(digests),
        tensors_equal_inputs=32, rejected_privileged_fields=rejects, process_checks=process_checks,
        resource_checks=8, logit_audit_cases=4, native_prefix_exact=True, simulator_entry_removed=True,
        gpu_context_created=False, model_forwards=0, optimizer_updates=0))
    shared = LINE / 'sft_acceptance/ordinary_history8_paired_train_r1'
    extra = [shared / n for n in ('supervise.py', 'runtime.py')]
    extra += [LINE / 'sft_acceptance/ordinary_prefix_history8_v1' / n for n in ('owned_process_r1.py', 'tests_r1.py')]
    extra += [LINE / 'closed_loop_bench/ordinary_stop_calibration_fit_v2/tree_size.py',
              source.INPUT / 'POLICY_INPUTS.jsonl', source.INPUT / 'SUPERVISION_ONLY.jsonl',
              source.FIT / 'run_001/MODEL_LOADED.json']
    for path in extra + list(HERE.glob('*.py')) + list(HERE.glob('*.json')) + [HERE / 'PLAN_ZH.md']:
        files[str(path)] = sha(path)
    write(HERE / 'SOURCE_LOCK.json', dict(unix=time.time(), files=files, protocol_sha256=sha(HERE / 'REPLAY_PROTOCOL.json')))
    common.verify_lock()
    write(HERE / 'PREPARATION_RESULT.json', dict(status='PASS_FROZEN_READY', unix=time.time(),
        source_files=len(files), source_lock_sha256=sha(HERE / 'SOURCE_LOCK.json'),
        replay_protocol_sha256=sha(HERE / 'REPLAY_PROTOCOL.json'), gpu_launches=0))
    print(json.dumps(read(HERE / 'PREPARATION_RESULT.json')), flush=True)


if __name__ == '__main__':
    main()
