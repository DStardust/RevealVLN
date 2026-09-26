"""CPU contracts plus two real FIT cached-feature optimizer updates; no GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from execution_memory import ActionOutcomeMemory, START, STOP, FORWARD, LEFT

torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1209)
        self.model = ActionOutcomeMemory(16)
        self.x = torch.randn(24, 16)
        self.actions = torch.full((24,), FORWARD, dtype=torch.long)
        self.actions[-1] = STOP
        self.queries = torch.tensor([0, 8, 16, 23])
        self.actor = torch.randn(4, 16)
        self.native = torch.randn(4, 4)

    def run_trace(self, features=None, actions=None):
        return self.model(self.x if features is None else features,
                          self.actions if actions is None else actions,
                          self.queries, self.actor, self.native)

    def test_actual_previous_action_changes_memory(self):
        memory = self.model.reset()
        a = self.model.update(self.x[:1], memory, self.x[1:2], torch.tensor([FORWARD]))
        b = self.model.update(self.x[:1], memory, self.x[1:2], torch.tensor([LEFT]))
        self.assertFalse(torch.equal(a, b))

    def test_feature_change_has_separate_write_path(self):
        memory, action = self.model.reset(), torch.tensor([FORWARD])
        a = self.model.update(self.x[:1], memory, self.x[1:2], action)
        b = self.model.update(self.x[:1], memory, self.x[2:3], action)
        self.assertFalse(torch.equal(a, b))
        with torch.no_grad():
            self.model.change_writer.weight.zero_()
        a = self.model.update(self.x[:1], memory, self.x[1:2], action)
        b = self.model.update(self.x[:1], memory, self.x[2:3], action)
        self.assertTrue(torch.equal(a, b))

    def test_future_cannot_change_earlier_states(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        F.cross_entropy(self.run_trace()['logits'], torch.tensor([0, 1, 2, 3])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        original = self.run_trace()
        self.assertGreater(float((original['logits'] - self.native).detach().abs().max()), 0)
        changed_x = self.x.clone()
        changed_x[9:] += 3 * torch.randn_like(changed_x[9:])
        changed_actions = self.actions.clone()
        changed_actions[9:-1] = LEFT
        changed = self.run_trace(changed_x, changed_actions)
        self.assertTrue(torch.equal(original['memory'][:9], changed['memory'][:9]))
        self.assertTrue(torch.equal(original['logits'][:2], changed['logits'][:2]))
        self.assertFalse(torch.equal(original['memory'][9:], changed['memory'][9:]))
        Contracts.causality_evidence = dict(actor_ce_warmup_updates=1,
            actor_residual_max_abs=float((original['logits'] - self.native).detach().abs().max()),
            future_changes_from_step=9, checked_query_steps=[0, 8],
            early_memory_bitwise_equal=True, early_query_logits_bitwise_equal=True,
            future_features_and_executed_actions_changed=True)

    def test_stop_is_not_a_new_observation(self):
        out = self.run_trace()
        self.assertEqual(len(out['memory']), len(self.actions))
        with self.assertRaisesRegex(ValueError, 'STOP_HAS_NO_NEXT_OBSERVATION'):
            self.model.update(self.x[:1], self.model.reset(), self.x[:1], torch.tensor([STOP]))
        bad = self.actions.clone()
        bad[10] = STOP
        with self.assertRaisesRegex(ValueError, 'OBSERVATION_AFTER_STOP'):
            self.run_trace(actions=bad)

    def test_start_requires_clean_reset(self):
        first = self.model.update(self.x[:1], self.model.reset(), None, torch.tensor([START]))
        with self.assertRaisesRegex(ValueError, 'START_REQUIRES_RESET'):
            self.model.update(self.x[:1], first, None, torch.tensor([START]))
        with self.assertRaisesRegex(ValueError, 'MOTION_REQUIRES_PREVIOUS'):
            self.model.update(self.x[:1], self.model.reset(), None, torch.tensor([FORWARD]))

    def test_episode_and_batch_reset_isolation(self):
        first = self.run_trace()['memory']
        self.run_trace(features=self.x + 10)
        self.assertTrue(torch.equal(first, self.run_trace()['memory']))
        memory = self.model.reset(2)
        memory[1] = first[5]
        out = self.model.update(self.x[:2], memory, self.x[2:4], torch.tensor([START, FORWARD]))
        expected = self.model.update(self.x[:1], self.model.reset(), None, torch.tensor([START]))
        torch.testing.assert_close(out[:1], expected)

    def test_zero_actor_exactly_preserves_native(self):
        self.assertTrue(torch.equal(self.run_trace()['logits'], self.native))

    def test_long_sequence_gradient_reaches_early_change(self):
        # Open the zero-initialized actor with an actual CE update first.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        F.cross_entropy(self.run_trace()['logits'], torch.tensor([0, 1, 2, 3])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        features = torch.randn(96, 16, requires_grad=True)
        actions = torch.full((96,), FORWARD, dtype=torch.long)
        actions[-1] = STOP
        delta_terms = []
        memory = self.model.reset()
        def record_change(_module, _inputs, output):
            output.retain_grad()
            delta_terms.append(output)
        handle = self.model.change_writer.register_forward_hook(record_change)
        for step in range(96):
            memory = self.model.update(features[step:step + 1], memory,
                None if step == 0 else features[step - 1:step],
                torch.tensor([START]) if step == 0 else actions[step - 1:step])
        handle.remove()
        logits = self.native[:1] + self.model.action_delta(self.actor[:1], memory)
        F.cross_entropy(logits, torch.tensor([LEFT])).backward()
        self.assertGreater(float(features.grad[0].norm()), 0)
        self.assertGreater(float(delta_terms[1].grad.norm()), 0)
        for module in (self.model.writer, self.model.change_writer, self.model.executed_action):
            self.assertGreater(float(module.weight.grad.norm()), 0)
        Contracts.gradient_evidence = dict(sequence_observations=96,
            supervised_query_step=95, early_feature_step=0,
            early_feature_gradient_l2=float(features.grad[0].norm()),
            early_change_step=1, early_change_output_gradient_l2=float(delta_terms[1].grad.norm()),
            writer_gradient_l2=float(self.model.writer.weight.grad.norm()),
            change_writer_gradient_l2=float(self.model.change_writer.weight.grad.norm()),
            previous_action_embedding_gradient_l2=float(self.model.executed_action.weight.grad.norm()),
            actor_warmup_updates=1, detach_used=False, scope='Synthetic CPU gradient contract, separate from real FIT evidence.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def real_fit_probe(base):
    run = base / 'recovery_action_v1/runs/action_001'
    pool_path = run / 'data/POOLS.pt'
    pack = torch.load(pool_path, map_location='cpu', weights_only=True, mmap=True)
    # Container includes FIT/DEV metadata; only this FIT row's tensor pages are
    # accessed. No DEV losses, labels, traces or unseen results are read.
    row = next(row for row in pack['rows'] if row['partition'] == 'FIT')
    manifest = json.loads((run / 'features/DATA_MANIFEST.json').read_text())
    entry = next(entry for entry in manifest['episodes'] if entry['id'] == row['id'])
    trace_path = Path(entry['trajectory'])
    trace = json.loads(trace_path.read_text())
    expected_hash = json.loads((run / 'features/SOURCE_LOCK.json').read_text())['files'][str(trace_path)]
    actual_hash = sha(trace_path)
    if actual_hash != expected_hash:
        raise ValueError('FIT_TRACE_SOURCE_HASH_MISMATCH')
    actions = torch.tensor(trace['actions'], dtype=torch.long)
    if len(actions) != len(row['memory_features']) or not torch.equal(actions[row['query_steps']], row['targets']):
        raise ValueError('FIT_ACTION_FEATURE_ALIGNMENT')
    if trace['query_steps'] != row['query_steps'].tolist() or actions[-1] != STOP or not trace['admitted']:
        raise ValueError('FIT_TRAJECTORY_CONTRACT')
    torch.manual_seed(1209)
    model = ActionOutcomeMemory(row['memory_features'].shape[-1]).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    parameter_count = sum(p.numel() for p in model.parameters())
    added_parameters = sum(p.numel() for p in model.change_writer.parameters()) + sum(p.numel() for p in model.executed_action.parameters())
    forward = lambda: model(row['memory_features'].float(), actions, row['query_steps'],
                            row['actor_features'].float(), row['base_logits'].float())
    with torch.no_grad():
        initial = forward()
        zero_actor_equal = torch.equal(initial['logits'], row['base_logits'].float())
    updates = []
    watched = ('writer.weight', 'change_writer.weight', 'executed_action.weight', 'actor.2.weight')
    for step in range(2):
        before = {name: value.detach().clone() for name, value in model.named_parameters() if name in watched}
        optimizer.zero_grad(set_to_none=True)
        output = forward()
        loss = F.cross_entropy(output['logits'][row['known']], row['targets'][row['known']])
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('NONFINITE_CPU_FIT_BACKWARD')
        gradients = {name: float(value.grad.norm()) if value.grad is not None else None
                     for name, value in model.named_parameters() if name in watched}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        changes = {name: float((value.detach() - before[name]).norm())
                   for name, value in model.named_parameters() if name in watched}
        updates.append(dict(step=step + 1, loss=float(loss.detach()), gradient_norms=gradients, parameter_change_l2=changes))
    for name in watched:
        if not updates[-1]['gradient_norms'][name] > 0 or not updates[-1]['parameter_change_l2'][name] > 0:
            raise ValueError('MISSING_REAL_FIT_GRADIENT_OR_UPDATE:' + name)
    if not zero_actor_equal:
        raise ValueError('ZERO_ACTOR_FAILED_TO_PRESERVE_NATIVE')
    return dict(status='REAL_FIT_CACHED_FEATURE_FORWARD_BACKWARD_UPDATE_COMPLETE', row_id=row['id'],
        partition=row['partition'], kind=row['kind'], house=row['house'],
        real_observations=len(actions), actual_motion_transitions=len(actions) - 1,
        query_count=len(row['query_steps']), supervised_queries=int(row['known'].sum()),
        actions_from_actual_trace=True, trace_path=str(trace_path), trace_sha256=actual_hash,
        pool_path=str(pool_path), pool_sha256_recorded=json.loads((run / 'data/ADMISSION.json').read_text())['pools_sha256'],
        pool_hash_recomputed=False, loaded_feature_width=row['memory_features'].shape[-1],
        feature_semantics='Mixed causal visual/instruction/previous-action feature; difference is not pure visual change.',
        action_information_overlap='Feature differences include changes in the already encoded previous action. The explicit action embedding can duplicate existing information; this is a trainable architecture adaptation, not novelty or pure visual-consequence evidence.',
        parameters_total=parameter_count, parameters_added=added_parameters,
        baseline_stored_parameters=parameter_count - added_parameters,
        inherited_unused_parameters=['state_reader', 'query', 'key'],
        new_action_embedding_rows=5, start_index=START, stop_index=STOP,
        stop_embedding_used=False, zero_actor_preserves_native_logits=zero_actor_equal,
        first_update_interpretation='Zero actor gives zero writer/change/action gradients on update 1; their small first-step changes are AdamW weight decay, not outcome learning. Update 2 has nonzero gradients and updates on all three paths.',
        updates=updates, optimizer_updates=2, weights_saved=False,
        frozen_backbone_loaded=False, backbone_updates=0,
        dev_tensors_accessed=False, unseen_accessed=False,
        fairness='Added writer projection and action embedding increase capacity; require same-capacity ordinary-writer control before attributing benefit.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'CPU_EXECUTION_MEMORY_RESULT.json')
    args = parser.parse_args()
    began = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    record = dict(status='CPU_CONTRACT_FAILED', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), gpu_hours=0,
                  torch_threads=torch.get_num_threads(), method_benefit='NOT_MEASURED')
    if result.wasSuccessful():
        record['future_causality_contract'] = Contracts.causality_evidence
        record['cross_step_gradient_contract'] = Contracts.gradient_evidence
        record['fit_probe'] = real_fit_probe(args.base)
        record['status'] = 'CPU_IMPLEMENTATION_AND_REAL_FIT_UPDATE_COMPLETE'
    record['wall_seconds'] = time.time() - began
    record['source_sha256'] = {str(path): sha(path) for path in
        (Path(__file__).resolve(), Path(__file__).resolve().with_name('execution_memory.py'))}
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
