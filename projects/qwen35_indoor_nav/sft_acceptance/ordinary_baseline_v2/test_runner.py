"""CPU stdlib schedule/admission counterexamples; no GPU/model imports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ordinary_runner_test_target', HERE / 'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
CONFIG = dict(epochs=3, chunk_steps=4, seed=1109, learning_rate=.0001,
              weight_decay=.01, gradient_clip=1., gradient_accumulation_chunks=8,
              schedule='constant', loss='unweighted_action_ce', max_sequence_tokens=1024)


class FakeRecord:
    def __init__(self, row):
        self.row = row

    def decision(self, t, decode_rgb=True):
        assert decode_rgb
        return dict(control=dict(decision_step=t, memory_reset=t == 0),
                    policy=dict(instruction=self.row['record_id'], images=[t],
                                executed_actions=['move_forward'] * min(t, 8)),
                    supervision=dict(target_action='STOP' if t == self.row['decisions'] - 1 else 'move_forward', ce_mask=True))


class FakeBackend:
    def __init__(self, accumulation=8):
        self.calls = []
        self.resets = self.detaches = self.pending = 0
        self.accumulation = accumulation

    def zero_memory(self):
        self.resets += 1
        return 0

    def detach_memory(self, memory):
        self.detaches += 1
        return memory

    def train_chunk(self, decisions, memory, finalize=False):
        assert memory == decisions[0]['control']['decision_step'], 'lost history'
        self.calls.append(copy.deepcopy(decisions))
        self.pending += 1
        update = finalize or self.pending == self.accumulation
        if update:
            self.pending = 0
        return memory + len(decisions), dict(optimizer_step=update)


class RunnerTests(unittest.TestCase):
    def rows(self):
        return [dict(record_id='route5', decisions=5), dict(record_id='route9', decisions=9)]

    def test_import_has_no_heavy_dependency(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        self.assertNotIn('q35n_readonly_recovery_policy', sys.modules)

    def test_full_epochs_partial_chunk_stop_and_memory(self):
        backend = FakeBackend()
        checkpoints = []
        cursor = r.train_epochs(self.rows(), CONFIG, backend, FakeRecord,
                                checkpoint=lambda c, m, s: checkpoints.append((c, m)))
        self.assertEqual(cursor['decisions'], 42)
        self.assertEqual(cursor['epoch'], 3)
        self.assertEqual(cursor['chunks'], 15)
        self.assertEqual(cursor['updates'], 3)  # five chunks/epoch, flush every epoch
        self.assertEqual(backend.resets, 6)
        self.assertEqual(backend.detaches, 15)
        self.assertEqual([c['epoch'] for c, m in checkpoints], [1, 2, 3])
        self.assertTrue(all(m is None for c, m in checkpoints))
        self.assertEqual(sum(d['supervision']['target_action'] == 'STOP' for call in backend.calls for d in call), 6)
        self.assertEqual(sum(len(call) == 1 for call in backend.calls), 6)

    def test_resume_epoch_does_not_repeat(self):
        full = FakeBackend()
        r.train_epochs(self.rows(), CONFIG, full, FakeRecord)
        one = FakeBackend()
        cursor = r.train_epochs(self.rows(), dict(CONFIG, epochs=1), one, FakeRecord)
        resumed = FakeBackend()
        end = r.train_epochs(self.rows(), CONFIG, resumed, FakeRecord, cursor)
        self.assertEqual(one.calls + resumed.calls, full.calls)
        self.assertEqual(end['decisions'], 42)

    def test_shuffle_once_per_epoch_and_progress_at_boundaries(self):
        progress = []
        with patch.object(r, 'route_order', wraps=r.route_order) as order:
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), FakeRecord,
                           progress=lambda c, m: progress.append(c))
        self.assertEqual(order.call_count, 3)
        self.assertEqual([c['epoch'] for c in progress], [1, 2, 3])

    def test_epoch_metrics_aggregate_every_chunk(self):
        class Measured(FakeBackend):
            def train_chunk(self, decisions, memory, finalize=False):
                memory, metrics = super().train_chunk(decisions, memory, finalize)
                confusion = [[0] * 4 for _ in range(4)]
                for decision in decisions:
                    a = r.ACTIONS.index(decision['supervision']['target_action'])
                    confusion[a][a] += 1
                return memory, dict(metrics, ce=2.0, confusion=confusion)
        reports = []
        r.train_epochs(self.rows(), CONFIG, Measured(), FakeRecord,
                       checkpoint=lambda c, m, s: reports.append(s))
        for report in reports:
            self.assertEqual(report['online_training_epoch_decisions'], 14)
            self.assertEqual(report['online_training_ce'], 2.0)
            self.assertEqual(report['online_training_confusion'][3][3], 2)
            self.assertEqual(report['online_training_stop_precision'], 1.0)
            self.assertFalse(report['evaluation_performed'])

    def test_accumulation_update_flushes_short_tail(self):
        backend = FakeBackend(2)
        cursor = r.train_epochs(self.rows(), dict(CONFIG, epochs=1, gradient_accumulation_chunks=2), backend, FakeRecord)
        self.assertEqual(cursor['updates'], 3)
        self.assertEqual(backend.pending, 0)

    def test_policy_leakage_rejected(self):
        class Bad(FakeRecord):
            def decision(self, *a, **kw):
                d = super().decision(*a, **kw)
                d['policy']['future_action'] = 'STOP'
                return d
        with self.assertRaisesRegex(ValueError, 'POLICY_WHITELIST'):
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), Bad)

    def test_early_stop_rejected(self):
        class Bad(FakeRecord):
            def decision(self, *a, **kw):
                d = super().decision(*a, **kw)
                d['supervision']['target_action'] = 'STOP'
                return d
        with self.assertRaisesRegex(ValueError, 'TERMINAL_STOP'):
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), Bad)

    def test_missing_terminal_stop_rejected(self):
        class Bad(FakeRecord):
            def decision(self, *a, **kw):
                d = super().decision(*a, **kw)
                d['supervision']['target_action'] = 'move_forward'
                return d
        with self.assertRaisesRegex(ValueError, 'TERMINAL_STOP'):
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), Bad)

    def test_memory_reset_on_middle_rejected(self):
        class Bad(FakeRecord):
            def decision(self, *a, **kw):
                d = super().decision(*a, **kw)
                d['control']['memory_reset'] = True
                return d
        with self.assertRaisesRegex(ValueError, 'MEMORY_RESET'):
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), Bad)

    def test_nonzero_resume_requires_history(self):
        cursor = dict(r.initial_cursor(), step=4)
        with self.assertRaisesRegex(ValueError, 'RESUME_MEMORY_REQUIRED'):
            r.train_epochs(self.rows(), CONFIG, FakeBackend(), FakeRecord, cursor)

    def test_preparation_rejects_before_file_read(self):
        with patch.object(r, 'read_json', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'GPU_RUN_NOT_ADMITTED'):
                r.admission(dict(status='PREPARATION_ONLY', gpu_run_allowed=False), {}, HERE / 'RUN_ADMISSION.json')

    def test_missing_admission_rejected_before_backend(self):
        with patch.object(r, 'preflight', return_value=({}, [], {})), \
                patch.object(r, 'TorchBackend', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'RUN_ADMISSION_REQUIRED'):
                r.main(['--protocol', str(HERE / 'PROTOCOL.json'), '--snapshot', str(HERE / 'snapshot_v1'), '--train'])

    def test_wrong_binding_rejected(self):
        report = {k: 'correct' for k in ('protocol_sha256', 'snapshot_result_sha256', 'snapshot_seal_sha256', 'training_index_sha256', 'code_sha256')}
        with patch.object(r, 'read_json', return_value={}):
            with self.assertRaisesRegex(ValueError, 'ADMISSION_BINDING_MISMATCH'):
                r.admission(dict(status='RUN_ADMITTED', gpu_run_allowed=True), report, HERE / 'RUN_ADMISSION.json')

    def test_config_ignores_no_alternative_loss(self):
        self.assertEqual(r.configuration(dict(training=CONFIG)), CONFIG)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_LOSS'):
            r.configuration(dict(training=dict(CONFIG, loss='crossed_bce')))
        with self.assertRaisesRegex(ValueError, 'INVALID_SEQUENCE'):
            r.configuration(dict(training=dict(CONFIG, max_sequence_tokens=0)))

    def test_scoped_rejects_outside_project(self):
        with self.assertRaisesRegex(ValueError, 'PATH_OUTSIDE_LINE'):
            r.scoped('/tmp/not_q35n')

    def preflight_fixture(self, *, counts=None, split=None, source_hash='ok'):
        snapshot = HERE / 'test_virtual_snapshot'
        base = r.LINE / 'data_pipeline/virtual_test_source'
        row = dict(sourceRoot=str(base.relative_to(r.ROOT)), policy_file='p.json',
                   supervision_file='s.json', rgb_reference_root='.', source='R2R',
                   scene_group='fit_house', physical_source_route_sha256='physical', decisions=5,
                   record_id='record', policy_sha256='ok', supervision_sha256='ok', split='FIT')
        seal = {name: 'ok' for name in ('RESULT.json', 'TRAINING_INDEX.jsonl', 'SOURCE_HASHES.json', 'SPLIT.json')}
        values = {
            HERE / 'PROTOCOL.json': dict(training=CONFIG, status='PREPARATION_ONLY', gpu_run_allowed=False),
            snapshot / 'RESULT.json': dict(training_index_sha256='ok', counts=counts or dict(
                instruction_records=1, instruction_conditioned_decisions=5, strict_routes=1)),
            snapshot / 'SEAL.json': seal,
            snapshot / 'SPLIT.json': split or dict(FIT=['fit_house'], INTERNAL_DEV=['dev'], INTERNAL_CONFIRM=['confirm']),
            snapshot / 'SOURCE_HASHES.json': {str((base / name).relative_to(r.ROOT)): source_hash for name in ('p.json', 's.json')},
        }
        with patch.object(r, 'verify_code', return_value={}), patch.object(r, 'sha256', return_value='ok'), \
                patch.object(r, 'read_json', side_effect=lambda path: values[Path(path)]), \
                patch.object(Path, 'read_text', return_value=json.dumps(row)):
            return r.preflight(HERE / 'PROTOCOL.json', snapshot)

    def test_preflight_counts_split_and_source_bindings(self):
        _, _, report = self.preflight_fixture()
        self.assertEqual(report['planned_decisions'], 15)
        self.assertFalse(report['formal_fullscale_launch_ready'])
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_COUNTS_MISMATCH'):
            self.preflight_fixture(counts=dict(instruction_records=2, instruction_conditioned_decisions=5, strict_routes=1))
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_SPLIT_OVERLAP'):
            self.preflight_fixture(split=dict(FIT=['fit_house'], INTERNAL_DEV=['fit_house'], INTERNAL_CONFIRM=[]))
        with self.assertRaisesRegex(ValueError, 'ROW_SOURCE_HASH_BINDING'):
            self.preflight_fixture(source_hash='changed')


if __name__ == '__main__':
    unittest.main()
