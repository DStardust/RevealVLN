"""Synthetic integrity/causality fixtures; no new capture or model efficacy claim."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import nn
from data import load_rows, row_logits, adapt_old_query_cache_for_cpu_smoke, _sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    root = Path(root)
    trajectory_path = root / 'source.json'
    trace = dict(id=7, house='FIT_HOUSE', partition='FIT', kind='RECOVERY', cutoff=4,
        actions=[1, 2, 1, 3, 2, 0], query_steps=[0, 4], rgb_sha256=list('abcdef'), admitted=True, success=True)
    write(trajectory_path, trace)
    entry = dict(id=7, house='FIT_HOUSE', partition='FIT', kind='RECOVERY',
        trajectory=str(trajectory_path), trajectory_sha256=_sha(trajectory_path))
    protocol = dict(planned_trajectories=1, expected_base_state_sha256='frozen_base')
    write(root / 'PROTOCOL.json', protocol)
    write(root / 'DATA_MANIFEST.json', dict(episodes=[entry]))
    write(root / 'SOURCE_LOCK.json', dict(files={str(path): _sha(path) for path in
        (root / 'PROTOCOL.json', root / 'DATA_MANIFEST.json', trajectory_path)}))
    session = root / 'capture/session_001'
    identity = dict(base_state_sha256='frozen_base', base_updates=0, hidden_size=2,
        source_lock_sha256=_sha(root / 'SOURCE_LOCK.json'), protocol_sha256=_sha(root / 'PROTOCOL.json'))
    write(session / 'RUNTIME_IDENTITY.json', identity)
    write(session / 'STATE_SEAL.json', dict(base_before='frozen_base', base_after='frozen_base',
        source_lock_sha256=_sha(root / 'SOURCE_LOCK.json'), complete_ids=[7]))
    group = session / 'episodes/7'
    group.mkdir(parents=True)
    (group / 'TRACE.jsonl').write_text('{"scope":"SYNTHETIC_CPU_FIXTURE"}\n')
    cache = dict(memory_features=torch.arange(12, dtype=torch.float32).reshape(6, 2) / 20,
        actor_features=torch.arange(12, dtype=torch.float32).reshape(6, 2) / 30,
        base_logits=torch.zeros(6, 4), executed_actions=torch.tensor(trace['actions']),
        query_steps=torch.tensor([0, 4]), actor_context_steps=torch.tensor([0, 0, 0, 0, 4, 4]),
        action_token_offsets=torch.tensor([0, 1, 2, 3, 0, 1]), supervision_region=torch.tensor([False] * 4 + [True] * 2),
        new_training_admission=False, id=7, house='FIT_HOUSE', partition='FIT', kind='RECOVERY', cutoff=4,
        source_trajectory_sha256=entry['trajectory_sha256'],
        representation='REAL_ALL_ACTION_TOKEN_CAPTURE_WITH_QUERY_START_MEMORY')
    torch.save(cache, group / 'CACHE.pt')
    receipt = dict(status='TRAJECTORY_RECAPTURE_ADMITTED', trajectory_id=7, split='FIT', kind='RECOVERY',
        physical_replay_exact=True, active_stop=True, runtime_identity_sha256=_sha(session / 'RUNTIME_IDENTITY.json'),
        cache_path=str(group / 'CACHE.pt'), cache_sha256=_sha(group / 'CACHE.pt'),
        trace_sha256=_sha(group / 'TRACE.jsonl'), physical_actions=6, captured_all_actor_rows=6, captured_all_queries=2)
    write(group / 'COMPLETE.json', receipt)
    return trace, cache, group


def replace_cache(group, cache):
    torch.save(cache, group / 'CACHE.pt')
    receipt = json.loads((group / 'COMPLETE.json').read_text())
    receipt['cache_sha256'] = _sha(group / 'CACHE.pt')
    write(group / 'COMPLETE.json', receipt)


class TinyCausalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.seen_actions = []

    def reset(self, batch_size=1):
        return self.scale.new_zeros(batch_size, 1)

    def update(self, feature, memory, previous_feature, previous_action):
        self.seen_actions.append(previous_action.detach().cpu().tolist())
        previous = torch.zeros_like(feature) if previous_feature is None else previous_feature
        return memory + self.scale * (feature.sum(1, keepdim=True) + previous.sum(1, keepdim=True)) + previous_action[:, None] / 100

    def action_delta(self, feature, memory):
        value = memory[:, 0] + feature[:, 0]
        return torch.stack([value, -value, value / 2, value * 0], -1)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trace, self.cache, self.group = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_cache_preserves_known_and_admission(self):
        row = load_rows(self.root)[0]
        self.assertEqual(row['id'], 7)
        self.assertEqual(row['known'].tolist(), [False] * 4 + [True] * 2)
        self.assertFalse(row['new_training_admission'])
        self.assertEqual(row['targets'].tolist(), self.trace['actions'])

    def test_unsealed_and_missing_group_rejected(self):
        seal = self.root / 'capture/session_001/STATE_SEAL.json'
        saved = seal.read_text()
        seal.unlink()
        with self.assertRaisesRegex(ValueError, 'UNSEALED'):
            load_rows(self.root)
        seal.write_text(saved)
        (self.group / 'COMPLETE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            load_rows(self.root)

    def test_later_or_stale_context_is_rejected(self):
        for invalid in (1, 0):
            cache = dict(self.cache, actor_context_steps=self.cache['actor_context_steps'].clone())
            position = 1 if invalid == 1 else 4
            cache['actor_context_steps'][position] = invalid
            replace_cache(self.group, cache)
            with self.assertRaisesRegex(ValueError, 'ACTOR_CONTEXT'):
                load_rows(self.root)

    def test_mask_action_split_and_nonfinite_are_rejected(self):
        variants = [dict(self.cache, supervision_region=torch.ones(6, dtype=torch.bool)),
                    dict(self.cache, executed_actions=torch.tensor([1, 2, 1, 3, 1, 0])),
                    dict(self.cache, partition='DEV'),
                    dict(self.cache, memory_features=torch.full((6, 2), float('nan')))]
        for cache in variants:
            replace_cache(self.group, cache)
            with self.assertRaises(ValueError):
                load_rows(self.root)

    def test_stop_does_not_add_a_memory_observation(self):
        cache = dict(self.cache, memory_features=torch.zeros(7, 2))
        replace_cache(self.group, cache)
        with self.assertRaisesRegex(ValueError, 'MEMORY_FEATURES_SHAPE'):
            load_rows(self.root)

    def test_changed_cache_or_source_is_rejected_before_loading(self):
        with (self.group / 'CACHE.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'CAPTURE_CACHE_CHANGED'):
            load_rows(self.root)
        replace_cache(self.group, self.cache)
        (self.root / 'source.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'PHYSICAL_SOURCE_CHANGED'):
            load_rows(self.root)

    def test_late_token_readout_ignores_later_physical_observation(self):
        row = load_rows(self.root)[0]
        head = TinyCausalHead()
        original = row_logits(head, row, 'cpu')
        changed = dict(row, memory_features=row['memory_features'].clone())
        changed['memory_features'][1:4] += 100
        revised = row_logits(head, changed, 'cpu')
        self.assertTrue(torch.equal(original[:4], revised[:4]))
        self.assertFalse(torch.equal(original[4:], revised[4:]))
        self.assertEqual(head.seen_actions[:6], [[4], [1], [2], [1], [3], [2]])
        self.assertNotEqual(float(original[1, 0].detach()), float(original[2, 0].detach()))

    def test_gradient_reaches_old_context_not_future_after_query(self):
        row = load_rows(self.root)[0]
        row = dict(row, memory_features=row['memory_features'].clone().requires_grad_())
        head = TinyCausalHead()
        row_logits(head, row, 'cpu')[-1, 0].backward()
        self.assertGreater(float(row['memory_features'].grad[0].abs().sum()), 0)
        self.assertEqual(float(row['memory_features'].grad[5].abs().sum()), 0)
        self.assertIsNotNone(head.scale.grad)

    def test_explicit_old_adapter_keeps_only_original_queries(self):
        old = dict(self.cache, actor_features=self.cache['actor_features'][[0, 4]],
            base_logits=self.cache['base_logits'][[0, 4]], targets=torch.tensor([1, 2]), known=torch.tensor([False, True]))
        row = adapt_old_query_cache_for_cpu_smoke(old, self.trace)
        self.assertEqual(row['missing_actor_positions'], 4)
        self.assertEqual(row['data_scope'], 'OLD_QUERY_ONLY_CPU_SMOKE')
        self.assertEqual(tuple(row_logits(TinyCausalHead(), row, 'cpu').shape), (2, 4))
        self.assertEqual(row['known'].tolist(), [False, True])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
