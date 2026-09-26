"""Real FIT, query-only CPU smoke; this is not all-token training or navigation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import train as training
from data import adapt_old_query_cache_for_cpu_smoke
from model import ExecutionAdaptation

torch.set_num_threads(2)
EVIDENCE = {}


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='cpu_training_', dir=HERE)
        cls.root = Path(cls.temporary.name)
        source = HERE.parent / 'recovery_action_v1/runs/action_001'
        pool_path = source / 'data/POOLS.pt'
        admission = training.u.read(source / 'data/ADMISSION.json')
        if training.u.sha(pool_path) != admission['pools_sha256']:
            raise ValueError('REAL_POOL_SOURCE_CHANGED')
        cls.pack = torch.load(pool_path, map_location='cpu', weights_only=True, mmap=True)
        manifest = {r['id']: r for r in training.u.read(source / 'features/DATA_MANIFEST.json')['episodes']}
        locked = training.u.read(source / 'features/SOURCE_LOCK.json')['files']
        cls.rows, cls.inputs, evidence = {}, {}, []
        for kind in ('RECOVERY', 'PRESERVATION'):
            old = next(r for r in cls.pack['rows'] if r['partition'] == 'FIT' and r['kind'] == kind)
            path = Path(manifest[old['id']]['trajectory'])
            digest = training.u.sha(path)
            if digest != locked[str(path)]:
                raise ValueError('REAL_PHYSICAL_TRACE_CHANGED')
            trace = training.u.read(path)
            row = adapt_old_query_cache_for_cpu_smoke(old, trace)
            cls.rows[row['id']] = row
            cls.inputs[str(path)] = digest
            evidence.append(dict(id=row['id'], kind=kind, partition=row['partition'], house=row['house'],
                physical_observations=len(row['memory_features']), actor_queries=len(row['targets']),
                supervised_queries=int(row['known'].sum()), unknown_queries=int((~row['known']).sum()),
                missing_actor_positions=row['missing_actor_positions'], trajectory=str(path), trajectory_sha256=digest))
        cls.recovery, cls.ordinary = cls.rows.values()
        torch.manual_seed(42)
        model = ExecutionAdaptation(3584, 'DELTA')
        old_initial = training.SOURCE / 'data/42/INITIAL.pt'
        initial_admission = training.u.read(old_initial.parent / 'ADMISSION.json')
        if training.u.sha(old_initial) != initial_admission['initial_sha256']:
            raise ValueError('SHARED_ORIGINAL_INITIAL_CHANGED')
        missing = model.load_state_dict(torch.load(old_initial, weights_only=True, map_location='cpu'), strict=False)
        if set(missing.missing_keys) != {'change_writer.weight', 'executed_action.weight'} or missing.unexpected_keys:
            raise ValueError('INITIAL_ARCHITECTURE_MISMATCH')
        cls.initial = cls.root / 'INITIAL_s42.pt'
        torch.save(model.state_dict(), cls.initial)
        cls.weights = cls.pack['class_weights']
        EVIDENCE.update(data_scope='OLD_QUERY_ONLY_CPU_SMOKE', pool_path=str(pool_path),
            pool_sha256=admission['pools_sha256'], real_fit_rows=evidence,
            original_initial_sha256=initial_admission['initial_sha256'], base_loaded=False,
            base_updates=0, gpu_used=False, cpu_threads=torch.get_num_threads(),
            dev_or_unseen_tensor_rows_used=0, navigation_efficacy_measured=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, name):
        run = self.root / name
        run.mkdir()
        config = dict(version='EXPLICIT_REAL_FIT_CPU_SMOKE', width=3584, seeds=[42],
            modes=['CURRENT', 'DELTA'], steps=3, checkpoint_every=1, device='cpu',
            optimizer=dict(lr=1e-4, weight_decay=.01, clip_grad_norm=1.),
            class_weights=self.weights.tolist(), schedules={'42': [[self.recovery['id'], self.ordinary['id']]] * 3},
            source_files={str(HERE / n): training.u.sha(HERE / n) for n in ('model.py', 'train.py', 'data.py')},
            input_files=self.inputs, initial_files={'42': dict(path=str(self.initial), sha256=training.u.sha(self.initial))},
            debug_only=True, action_scope='OLD_QUERY_ONLY_CPU_SMOKE')
        training.u.write(run / 'TRAINING_CONFIG.json', config)
        return run, config

    def new_model(self, mode='DELTA'):
        model = ExecutionAdaptation(3584, mode)
        model.load_state_dict(torch.load(self.initial, weights_only=True, map_location='cpu'))
        return model

    def test_shared_initialization_and_real_fit_scope(self):
        left, right = self.new_model('CURRENT'), self.new_model('DELTA')
        self.assertTrue(equal_tree(left.state_dict(), right.state_dict()))
        self.assertEqual(sum(p.numel() for p in left.parameters()), sum(p.numel() for p in right.parameters()))
        self.assertTrue(all(r['partition'] == 'FIT' and r['data_scope'] == 'OLD_QUERY_ONLY_CPU_SMOKE' for r in self.rows.values()))
        self.assertFalse(any(r['new_training_admission'] for r in self.rows.values()))
        EVIDENCE['shared_initialization_equal'] = True
        EVIDENCE['parameters_per_arm'] = sum(p.numel() for p in left.parameters())

    def test_unknown_targets_do_not_enter_loss_or_gradient(self):
        model = self.new_model()
        self.assertTrue(bool((~self.recovery['known']).any()))
        loss, info = training.objective(model, self.recovery, self.ordinary, self.weights, 'cpu')
        loss.backward()
        expected = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
        masked = dict(self.recovery, targets=self.recovery['targets'].clone())
        masked['targets'][~masked['known']] = 999
        model.zero_grad(set_to_none=True)
        changed, changed_info = training.objective(model, masked, self.ordinary, self.weights, 'cpu')
        changed.backward()
        self.assertTrue(torch.equal(loss, changed))
        self.assertEqual(info, changed_info)
        self.assertTrue(equal_tree(expected, {n: p.grad for n, p in model.named_parameters()}))
        self.assertTrue(torch.equal(masked['executed_actions'], self.recovery['executed_actions']))
        EVIDENCE['unknown_label_loss_and_gradient_invariant'] = True

    def test_dev_and_invalid_masks_are_rejected_before_forward(self):
        model = self.new_model()
        for recovery, ordinary in [(dict(self.recovery, partition='DEV'), self.ordinary),
                                   (self.recovery, dict(self.ordinary, partition='DEV'))]:
            with self.assertRaisesRegex(ValueError, 'FIT_ONLY_OPTIMIZATION'):
                training.objective(model, recovery, ordinary, self.weights, 'cpu')
        for recovery, ordinary in [(dict(self.recovery, known=torch.zeros_like(self.recovery['known'])), self.ordinary),
                                   (self.recovery, dict(self.ordinary, known=torch.zeros_like(self.ordinary['known'])))]:
            with self.assertRaisesRegex(ValueError, 'INVALID_SUPERVISION_MASK'):
                training.objective(model, recovery, ordinary, self.weights, 'cpu')
        EVIDENCE['dev_and_unknown_mask_rejections'] = True

    def test_resume_exact_model_optimizer_rng_and_receipt_recovery(self):
        proofs = {}
        for mode in ('CURRENT', 'DELTA'):
            whole, config = self.config(mode + '_whole')
            split, split_config = self.config(mode + '_split')
            complete = training.train_one(whole, config, self.rows, 42, mode)
            paused = training.train_one(split, split_config, self.rows, 42, mode, stop_after=1)
            self.assertEqual(paused['status'], 'PAUSED_AT_CHECKPOINT')
            self.assertEqual(paused['completed_steps'], 1)
            one = torch.load(split / f'{mode}_s42/checkpoint_000001.pt', weights_only=False, map_location='cpu')
            self.assertEqual(one['schedule_cursor'], 1)
            self.assertTrue(one['optimizer']['state'])
            self.assertEqual(set(one['rng']), {'python', 'numpy', 'torch', 'cuda'})
            resumed = training.train_one(split, split_config, self.rows, 42, mode, resume=True)
            self.assertEqual(complete['status'], resumed['status'], 'COMPLETE')
            self.assertEqual(resumed['completed_steps'], 3)
            self.assertEqual(resumed['action_scope'], 'OLD_QUERY_ONLY_CPU_SMOKE')
            folder = split / f'{mode}_s42'
            a = torch.load(whole / f'{mode}_s42/FINAL.pt', weights_only=False, map_location='cpu')
            b = torch.load(folder / 'FINAL.pt', weights_only=False, map_location='cpu')
            for key in ('model', 'optimizer', 'rng', 'step', 'schedule_cursor'):
                self.assertTrue(equal_tree(a[key], b[key]), mode + ':' + key)
            logs = [json.loads(line) for path in sorted(folder.glob('STEPS_*.jsonl')) for line in path.read_text().splitlines()]
            self.assertEqual([r['step'] for r in logs], [1, 2, 3])
            for name in ('writer.weight', 'change_writer.weight', 'executed_action.weight'):
                self.assertGreater(logs[-1]['writer_gradients'][name], 0, mode + ':' + name)
            digest, old_logs = training.u.sha(folder / 'FINAL.pt'), set(folder.glob('STEPS_*.jsonl'))
            (folder / 'RESULT.json').unlink()
            rebuilt = training.train_one(split, split_config, self.rows, 42, mode, resume=True)
            self.assertEqual(rebuilt['status'], 'COMPLETE')
            self.assertEqual(training.u.sha(folder / 'FINAL.pt'), digest)
            self.assertEqual(set(folder.glob('STEPS_*.jsonl')), old_logs)
            with self.assertRaisesRegex(ValueError, 'RESUME_REQUIRED'):
                training.train_one(split, split_config, self.rows, 42, mode)
            proofs[mode] = dict(uninterrupted_steps=3, resumed_steps=[1, 2],
                model_optimizer_rng_exact=True, final_receipt_recovered_without_update=True,
                final_writer_gradients=logs[-1]['writer_gradients'],
                changed_parameters=resumed['changed_parameters'])
        EVIDENCE['resume'] = proofs
        EVIDENCE['actual_optimizer_steps'] = 12

    def test_bound_input_change_refuses_resume(self):
        run, config = self.config('input_change')
        marker = self.root / 'owned_input.txt'
        marker.write_text('recorded')
        config = copy.deepcopy(config)
        config['input_files'][str(marker)] = training.u.sha(marker)
        training.u.write(run / 'TRAINING_CONFIG.json', config)
        marker.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'TRAINING_INPUT_CHANGED'):
            training.train_one(run, config, self.rows, 42, 'DELTA', resume=True)
        EVIDENCE['input_hash_change_rejected'] = True


if __name__ == '__main__':
    began = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingContracts))
    EVIDENCE.update(passed=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), elapsed_seconds=time.monotonic() - began,
        source_sha256={str(HERE / n): hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                       for n in ('test_training.py', 'train.py', 'model.py', 'data.py')})
    print(json.dumps(EVIDENCE, indent=2, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
