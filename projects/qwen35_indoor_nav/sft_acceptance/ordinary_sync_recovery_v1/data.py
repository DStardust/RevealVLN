"""Ordinary baseline v3 data: decision-level samples, length bucketing, DDP sharding.

Read-only over the frozen ordinary_baseline_v2 snapshot. No memory state crosses
decisions: each sample is causally self-contained (instruction + <=2 recent RGB +
<=8 executed actions -> 4-way target). Routes only serve bucketing efficiency and
the FIT/DEV split; no future information enters policy inputs (sealed adapter).
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
REF = LINE / 'sft_acceptance/ordinary_baseline_v2'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']
# 224x224 grid 16x16 with 2x2 merge -> 64 image tokens; raw text holds 1 placeholder.
IMAGE_TOKEN_EXPANSION = 63


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows():
    """Sealed preflight: verifies snapshot seals and row bindings (CPU, read-only)."""
    runner = load_module('q35n_v3_reference_runner', REF / 'runner.py')
    protocol, rows, report = runner.preflight(REF / 'PROTOCOL.json', REF / 'snapshot_v1')
    return rows, report


def inflection_weight(actions, t, coef):
    """VLN-CE inflection weighting: step 0 and every action-change step get coef."""
    return float(coef) if (t == 0 or actions[t] != actions[t - 1]) else 1.0


def build_sample_index(rows, processor, coef, out_path):
    """One-time deterministic CPU build: per-decision (record, t, target, weight, est).

    est tokens = instruction template tokens + image expansion + executed history + 1
    action-query token. Two blank frames measure the fixed expansion exactly as the
    sealed TOKEN_AUDIT did (126 tokens for two 224x224 frames).
    """
    from PIL import Image
    require(not Path(out_path).exists(), 'SAMPLE_INDEX_EXISTS')

    def prompt(instruction, n):
        return processor.apply_chat_template([{'role': 'user', 'content': [
            *[{'type': 'image'} for _ in range(n)], {'type': 'text', 'text': instruction}]}],
            tokenize=False, add_generation_prompt=True)

    blank = Image.new('RGB', (224, 224))
    probe_text = prompt('layout check.', 2)
    raw = len(processor.tokenizer(probe_text)['input_ids'])
    expansion = processor(text=[probe_text], images=[blank, blank], return_tensors='pt')['input_ids'].shape[1] - raw
    require(expansion == 2 * IMAGE_TOKEN_EXPANSION, 'IMAGE_EXPANSION_CHANGED:%d' % expansion)

    root = LINE.parents[1]
    texts = []
    for row in rows:
        blob = (root / row['sourceRoot'] / row['policy_file']).read_bytes()
        texts.append(prompt(json.loads(blob)['instruction'], 2))
    encoded = processor.tokenizer(texts, padding=False, truncation=False)['input_ids']
    data_module = load_module('q35n_v3_sealed_data', REF / 'data.py')
    count = 0
    with Path(out_path).open('x') as stream:
        for record_idx, (row, text_ids) in enumerate(zip(rows, encoded)):
            source = root / row['sourceRoot']
            policy_blob = (source / row['policy_file']).read_bytes()
            supervision_blob = (source / row['supervision_file']).read_bytes()
            require(hashlib.sha256(policy_blob).hexdigest() == row['policy_sha256'], 'POLICY_HASH')
            require(hashlib.sha256(supervision_blob).hexdigest() == row['supervision_sha256'],
                    'SUPERVISION_HASH')
            actions = tuple(data_module.normalize_action(a)
                            for a in json.loads(supervision_blob)['actions'])
            require(len(actions) == row['decisions'], 'LENGTH_MISMATCH')
            require(actions[-1] == 'STOP' and 'STOP' not in actions[:-1], 'TERMINAL_STOP')
            n = len(actions)
            base = len(text_ids)
            for t in range(n):
                # t==0 carries one image (63 expansion), later steps two (126).
                est = base + (IMAGE_TOKEN_EXPANSION if t == 0 else 2 * IMAGE_TOKEN_EXPANSION) \
                    + min(t, 8) + 1
                entry = [record_idx, t, ACTIONS.index(actions[t]),
                         inflection_weight(actions, t, coef), est]
                stream.write(json.dumps(entry) + '\n')
                count += 1
            if record_idx % 2000 == 0:
                print(json.dumps(dict(records_indexed=record_idx)), flush=True)
        stream.flush()
    require(count == sum(row['decisions'] for row in rows), 'SAMPLE_COUNT_MISMATCH')
    return dict(samples=count, sha256=sha256(out_path), image_expansion_per_frame=IMAGE_TOKEN_EXPANSION,
                inflection_coef=coef)


def load_sample_index(path, expected_sha256, expected_count):
    path = Path(path)
    require(sha256(path) == expected_sha256, 'SAMPLE_INDEX_HASH')
    samples = []
    for line in path.read_text().splitlines():
        if line.strip():
            record_idx, t, target, weight, est = json.loads(line)
            samples.append(dict(record_idx=record_idx, t=t, target=target, weight=weight, est=est))
    require(len(samples) == expected_count, 'SAMPLE_INDEX_COUNT')
    return samples


def plan_epoch_batches(samples, max_tokens, seed, epoch, world_size):
    """Length-sorted token-budget packing, block-shuffled, tail dropped to a
    world_size multiple. Deterministic given (seed, epoch); the sort tie-break is
    seeded per epoch so the dropped tail set varies across epochs. Batches stay
    length-homogeneous because packing runs on the sorted order. Returns the
    rank-sharded batch lists (interleaved by position)."""
    order = list(range(len(samples)))
    tie = random.Random(seed * 1000003 + epoch)
    tie.shuffle(order)
    order.sort(key=lambda i: samples[i]['est'])  # stable: keeps seeded tie order
    batches = []
    current, current_tokens = [], 0
    for i in order:
        est = samples[i]['est']
        if current and current_tokens + est > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += est
    if current:
        batches.append(current)
    random.Random(seed * 9176 + epoch + 1).shuffle(batches)
    usable = len(batches) - (len(batches) % world_size)
    batches = batches[:usable]
    return [[batches[i] for i in range(rank, len(batches), world_size)] for rank in range(world_size)]


def advance_epoch_boundary(cursor, shard_len):
    """Epoch-boundary accounting: returns the updated cursor. Advances epoch and
    resets position iff the shard is exhausted; otherwise unchanged (mid-shard
    stop stays resumable). Pure function; unit-tested."""
    cursor = dict(cursor)
    if cursor['position'] >= shard_len:
        cursor['epoch'] += 1
        cursor['position'] = 0
    return cursor


class SampleStore:
    """Per-process lazily parsed records; RGB decoded per access via the sealed adapter."""

    def __init__(self, rows):
        self._rows = rows
        self._cache = {}
        self._data = load_module('q35n_v3_runtime_data', REF / 'data.py')

    def get(self, record_idx, t):
        if record_idx not in self._cache:
            self._cache[record_idx] = self._data.OrdinaryRecord(self._rows[record_idx])
        record = self._cache[record_idx]
        decision = record.decision(t)  # sealed: causal whitelist, pixel-hash-verified RGB
        require(decision['control']['decision_step'] == t, 'NONCAUSAL_STEP')
        policy = decision['policy']
        require(set(policy) == {'instruction', 'images', 'executed_actions'}, 'MODEL_WHITELIST')
        return dict(instruction=policy['instruction'], images=policy['images'],
                    executed=policy['executed_actions'],
                    target=ACTIONS.index(decision['supervision']['target_action']))
