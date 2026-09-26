"""CPU state/Transformers-processor contracts, not a live backbone evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from model import ExecutionAdaptation, START
from runtime import FeatureMemoryState, ChunkActionProcessor, make_logits_processors, selected_action
from data import row_logits


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1209)

    def head(self, mode):
        head = ExecutionAdaptation(16, mode).eval()
        with torch.no_grad():
            head.actor[-1].weight.normal_(0, .03)
        return head

    def test_sequence_and_live_all_token_logits_exact(self):
        features = torch.randn(13, 16)
        actions = torch.tensor([1, 2, 3] * 4 + [0])
        actors = torch.randn(13, 16)
        native = torch.randn(13, 4)
        for mode in ('DELTA', 'CURRENT'):
            head = self.head(mode)
            with torch.no_grad():
                expected = row_logits(head, dict(memory_features=features, executed_actions=actions,
                    actor_features=actors, base_logits=native,
                    actor_context_steps=torch.tensor([step // 4 * 4 for step in range(13)])), 'cpu')
            state = FeatureMemoryState(head)
            actual = []
            for step in range(13):
                state.observe(features[step], None if step == 0 else int(actions[step-1]))
                if step % 4 == 0:
                    state.begin_query()
                    snapshot, writes = state.memory.clone(), state.writes
                    for offset in range(min(4, 13-step)):
                        q = step + offset
                        actual.append(native[q:q+1] + state.delta(actors[q:q+1]))
                    self.assertTrue(torch.equal(snapshot, state.memory))
                    self.assertEqual(writes, state.writes)
                    state.end_query()
            self.assertTrue(torch.equal(expected, torch.cat(actual)), mode)
            self.assertGreater(float((torch.cat(actual) - native).abs().max()), 0)

    def test_reset_stop_and_generated_chunk_cannot_write(self):
        state = FeatureMemoryState(self.head('DELTA'))
        x = torch.randn(16)
        state.observe(x, None)
        original = state.memory.clone()
        state.begin_query()
        with self.assertRaisesRegex(ValueError, 'PHYSICAL_OBSERVATION_DURING_GENERATION'):
            state.observe(torch.randn(16), 2)
        with self.assertRaisesRegex(ValueError, 'EXECUTION_DURING_GENERATION'):
            state.mark_stopped()
        state.delta(torch.randn(1, 16))
        self.assertTrue(torch.equal(original, state.memory))
        state.end_query()
        state.observe(torch.randn(16), 3)
        self.assertEqual(state.previous_action, 3)
        state.mark_stopped()
        for action in (None, 0, 1):
            with self.assertRaisesRegex(ValueError, 'STOP_MUST_NOT_CREATE_OBSERVATION'):
                state.observe(x, action)
        self.assertEqual(state.writes, 2)
        state.reset()
        state.observe(x, None)
        self.assertTrue(torch.equal(original, state.memory))
        self.assertEqual(state.writes, 1)
        self.assertIsNone(state.previous_action)

    def test_real_transformers_processor_all_offsets_and_native_unchanged(self):
        state = FeatureMemoryState(self.head('CURRENT'))
        state.observe(torch.randn(16), None)
        state.begin_query()
        actor = torch.randn(1, 16)
        processor = ChunkActionProcessor(state, [8, 9], [0, 1, 2, 3], lambda: actor)
        processors = make_logits_processors(processor)
        scores = torch.full((1, 12), -100.)
        scores[:, :4] = torch.tensor([4., 1., 0., -1.])
        pristine = scores.clone()
        self.assertTrue(torch.equal(processors(torch.tensor([[10]]), scores), scores))
        self.assertTrue(torch.equal(processors(torch.tensor([[10, 8]]), scores), scores))
        ids = torch.tensor([[10, 8, 9]])
        for offset in range(4):
            result = processors(ids, scores)
            self.assertTrue(torch.equal(scores, pristine))
            self.assertTrue(torch.equal(result[:, 4:], scores[:, 4:]))
            self.assertTrue(torch.equal(result[:, :4], scores[:, :4] + state.delta(actor)))
            self.assertEqual(processor.records[-1]['offset'], offset)
            ids = torch.cat((ids, result.argmax(-1)[:, None]), dim=1)
        self.assertEqual(state.writes, 1)
        self.assertEqual(len({r['query_memory_sha256'] for r in processor.records}), 1)
        with self.assertRaisesRegex(ValueError, 'UNREGISTERED_FIFTH_ACTION_TOKEN'):
            processors(ids, scores)
        state.end_query()

    def test_argmax_no_stop_guard_and_nonfinite_rejected(self):
        self.assertEqual(int(selected_action(torch.tensor([0., 2., 2., 1.]))), 1)
        state = FeatureMemoryState(self.head('CURRENT'))
        with torch.no_grad():
            state.head.actor[-1].weight.zero_()
            state.head.actor[-1].bias.copy_(torch.tensor([-9., 9., 0., 0.]))
        state.observe(torch.randn(16), None)
        state.begin_query()
        p = ChunkActionProcessor(state, [8], [0, 1, 2, 3], lambda: torch.zeros(1, 16))
        scores = torch.full((1, 12), -100.)
        scores[:, :4] = torch.tensor([4., 1., 0., -1.])
        p(torch.tensor([[10]]), scores)
        revised = p(torch.tensor([[10, 8]]), scores)
        self.assertEqual(p.records[0]['native_action'], 0)
        self.assertEqual(p.records[0]['method_action'], 1)
        self.assertEqual(int(selected_action(revised[:, :4])[0]), 1)
        with self.assertRaisesRegex(ValueError, 'NONFINITE'):
            selected_action(torch.tensor([float('nan'), 0., 1., 2.]))

    def test_checkpoint_loader_binds_actual_final_recipe(self):
        from evaluate_worker import load_registered_head
        import common as u
        with tempfile.TemporaryDirectory(prefix='.runtime-head-', dir=Path(__file__).resolve().parent) as name:
            root = Path(name)
            config = dict(width=16, action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY',
                          base_updates=0, source_files={}, debug_only=False, steps=3,
                          seeds=[42], modes=['DELTA', 'CURRENT'])
            u.write(root / 'TRAINING_CONFIG.json', config)
            path = root / 'DELTA_s42/FINAL.pt'
            path.parent.mkdir()
            head = self.head('DELTA')
            torch.save(dict(model=head.state_dict(), step=3,
                binding=dict(seed=42, mode='DELTA', config_sha256=u.sha(root / 'TRAINING_CONFIG.json'))), path)
            record = dict(path=str(path), sha256=u.sha(path), seed=42, mode='DELTA',
                          training_config_sha256=u.sha(root / 'TRAINING_CONFIG.json'))
            loaded = load_registered_head(record, dict(steps=3), 'cpu')
            self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in head.state_dict().items()))
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            with self.assertRaisesRegex(ValueError, 'HEAD_TRAINING_BINDING_CHANGED'):
                load_registered_head(dict(record, mode='CURRENT'), dict(steps=3), 'cpu')
            with self.assertRaisesRegex(ValueError, 'HEAD_FILE_CHANGED'):
                load_registered_head(dict(record, sha256='bad'), dict(steps=3), 'cpu')
            with self.assertRaisesRegex(ValueError, 'REGISTERED_TRAINING_CONFIG_CHANGED'):
                load_registered_head(dict(record, training_config_sha256='bad'), dict(steps=3), 'cpu')
            config['debug_only'] = True
            u.write(root / 'TRAINING_CONFIG.json', config)
            digest = u.sha(root / 'TRAINING_CONFIG.json')
            torch.save(dict(model=head.state_dict(), step=3,
                binding=dict(seed=42, mode='DELTA', config_sha256=digest)), path)
            with self.assertRaisesRegex(ValueError, 'DEBUG_OR_NONFINAL_HEAD_REJECTED'):
                load_registered_head(dict(record, sha256=u.sha(path), training_config_sha256=digest), dict(steps=3), 'cpu')


if __name__ == '__main__':
    began = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    print(json.dumps(dict(status='CPU_RUNTIME_CONTRACT_PASS' if result.wasSuccessful() else 'CPU_RUNTIME_CONTRACT_FAIL',
                         tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                         seconds=time.time()-began, gpu_hours=0, live_backbone_forward_tested=False,
                         navigation_benefit='NOT_MEASURED')))
    raise SystemExit(not result.wasSuccessful())
