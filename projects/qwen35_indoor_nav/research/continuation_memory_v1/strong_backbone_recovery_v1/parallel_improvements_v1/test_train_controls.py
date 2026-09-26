"""CPU tests use two real FIT histories; they make no navigation claim."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import train_controls as t
from confirmation_model import sequence_logits


def assert_nested_equal(case, a, b):
    if isinstance(a, torch.Tensor):
        case.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        case.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        case.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_nested_equal(case, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        case.assertEqual(len(a), len(b))
        for left, right in zip(a, b):
            assert_nested_equal(case, left, right)
    else:
        case.assertEqual(a, b)


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='.q35n-control-cpu-', dir=HERE)
        cls.root = Path(cls.temp.name)
        cls.config, cls.pack = t.prepare(cls.root / 'continuous', [42], list(t.MODES), 3, 'cpu', debug_two_rows=True)
        cls.pair = cls.config['schedules']['42'][0]
        cls.rows = [cls.pack['rows'][i] for i in cls.pair]
        cls.evidence = dict(actual_cpu_updates=0, modes={}, rows=[dict(id=r['id'], house=r['house'],
                            partition=r['partition'], kind=r['kind'], length=len(r['memory_features']),
                            known_queries=int(r['known'].sum())) for r in cls.rows])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_resume_is_exact_for_all_controls(self):
        resumed_config, _ = t.prepare(self.root / 'resumed', [42], list(t.MODES), 3, 'cpu', debug_two_rows=True)
        for mode in t.MODES:
            complete = t.train_one(self.root / 'continuous', self.config, self.pack, 42, mode)
            paused = t.train_one(self.root / 'resumed', resumed_config, self.pack, 42, mode, stop_after=1)
            self.assertEqual(paused['status'], 'PAUSED_AT_CHECKPOINT')
            self.assertEqual(paused['completed_steps'], 1)
            with self.assertRaisesRegex(ValueError, 'EXISTING_MODEL_REQUIRES_RESUME'):
                t.train_one(self.root / 'resumed', resumed_config, self.pack, 42, mode)
            # Deliberately disturb all CPU RNGs before restore.
            random.random(); np.random.random(10); torch.rand(17)
            resumed = t.train_one(self.root / 'resumed', resumed_config, self.pack, 42, mode, resume=True)
            a = torch.load(self.root / 'continuous' / f'{mode}_s42/FINAL.pt', weights_only=False)
            b = torch.load(self.root / 'resumed' / f'{mode}_s42/FINAL.pt', weights_only=False)
            for key in ('model', 'optimizer', 'rng', 'step', 'schedule_cursor'):
                assert_nested_equal(self, a[key], b[key])
            self.assertEqual(complete['completed_steps'], resumed['completed_steps'])
            self.assertTrue(resumed['debug_only'])
            self.assertFalse(resumed['navigation_efficacy_measured'])
            self.assertIn('actor.2.weight', resumed['changed_parameters'])
            self.assertIn('writer.weight', resumed['changed_parameters'])
            # Recover even if the process ended immediately after FINAL.
            (self.root / 'resumed' / f'{mode}_s42/RESULT.json').unlink()
            sealed = t.train_one(self.root / 'resumed', resumed_config, self.pack, 42, mode, resume=True)
            self.assertEqual(sealed['status'], 'COMPLETE')
            self.evidence['actual_cpu_updates'] += 6
            self.evidence['modes'][mode] = dict(resume_model_optimizer_rng_bitwise_equal=True,
                                              complete_steps=3, resume_segments=[1, 2],
                                              changed_parameters=resumed['changed_parameters'])

    def test_known_mask_and_old_event_gradient(self):
        recovery, ordinary = self.rows
        self.assertTrue(bool((~recovery['known']).any()))
        self.assertTrue(bool((recovery['targets'][recovery['known']] == 0).any()))
        self.assertTrue(bool(ordinary['known'].all()))
        for mode in t.MODES:
            torch.manual_seed(42)
            model = t.UnitReadoutMemory(self.config['width'], mode)
            with torch.no_grad():
                model.actor[-1].weight.normal_(0, .01)
            altered = dict(recovery, targets=recovery['targets'].clone())
            altered['targets'][~recovery['known']] = 999
            before, info = t.objective(model, recovery, ordinary, self.pack['class_weights'])
            after, _ = t.objective(model, altered, ordinary, self.pack['class_weights'])
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(info['recovery_queries'], int(recovery['known'].sum()))
            observed = dict(recovery, memory_features=recovery['memory_features'].clone().requires_grad_(True))
            last = sequence_logits(model, observed)[-1, 0]
            last.backward()
            grad = observed['memory_features'].grad
            self.assertEqual(bool(grad[0].abs().sum() > 0), mode != 'LOCAL')
            self.assertGreater(float(grad[int(observed['query_steps'][-1])].norm()), 0)
        self.evidence['known_mask_excludes_unlabeled_prefix_queries'] = True
        self.evidence['teacher_STOP_has_known_query_target'] = True
        self.evidence['non_first_STOP_coverage_repaired'] = False
        self.evidence['full_unroll_old_feature_gradient_checked'] = True

    def test_binding_and_input_validation_fail_closed(self):
        row = copy.copy(self.rows[0])
        row['partition'] = 'DEV'
        with self.assertRaisesRegex(ValueError, 'TRAINING_REQUIRES_FIT'):
            t.validate_row(row, self.config['width'])
        row = copy.copy(self.rows[0])
        row['query_steps'] = row['query_steps'].clone()
        row['query_steps'][-1] = len(row['memory_features'])
        with self.assertRaisesRegex(ValueError, 'QUERY_OUTSIDE_HISTORY'):
            t.validate_row(row, self.config['width'])
        with self.assertRaisesRegex(ValueError, 'RUN_BINDING_CHANGED'):
            t.prepare(self.root / 'continuous', [42], list(t.MODES), 2, 'cpu', debug_two_rows=True)
        path = self.root / 'continuous/TRAINING_CONFIG.json'
        original = path.read_text()
        mutated = copy.deepcopy(self.config)
        mutated['source_files'][str(t.Path(t.__file__))] = 'source changed'
        t.u.write(path, mutated)
        try:
            with self.assertRaisesRegex(ValueError, 'RUN_BINDING_CHANGED'):
                t.prepare(self.root / 'continuous', [42], list(t.MODES), 3, 'cpu', debug_two_rows=True)
        finally:
            path.write_text(original)
        self.evidence['changed_source_config_rejected'] = True
        self.evidence['dev_training_and_bad_query_indices_rejected'] = True


if __name__ == '__main__':
    began = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainerTests))
    report = dict(status='CPU_TRAINER_IMPLEMENTATION_TEST_PASS' if result.wasSuccessful() else 'CPU_TEST_FAILED',
                  tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  cpu_seconds=time.time() - began, gpu_hours=0, backbone_loaded=False, base_updates=0,
                  dev_or_unseen_used=False, weights_retained=False,
                  evidence=getattr(TrainerTests, 'evidence', {}),
                  source_files=t.source_identity(), test_source_sha256=t.u.sha(Path(__file__)),
                  limitations=['No navigation efficacy measurement', 'Query-first only; missing later-token STOP features unchanged'])
    output = HERE / 'CPU_TRAINER_TEST_RESULT.json'
    if output.exists():
        raise FileExistsError('Preserve the existing CPU test receipt; use unittest for a rerun')
    t.u.write(output, report)
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
