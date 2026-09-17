"""CPU mock orchestration tests: no Habitat, models, GPU or file outputs."""
import collections
import copy
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('q35n_test_family_job', Path(__file__).with_name('family_job.py'))
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
from core_bridge import BudgetLedger, FreezeLedger


class FakeFactory:
    mode = 'pass'
    configurations = None
    def __init__(self, backend, compiler, budget, emit, task_a, task_b, context):
        self.budget, self.emit = budget, emit
        self.runner = types.SimpleNamespace(counts=collections.Counter())
    def trace(self, actions, seed=1109, complete=True):
        for a in actions:
            if a != 'S':
                self.budget.reserve_action()
                self.emit('action_completed', {'action': a})
            else:
                self.runner.counts['executed_stops'] += 1
        self.emit('trace', {'seed': seed, 'actions': actions, 'complete': complete, 'collisions': 0})
    def discover(self, configs, ledger, bundle):
        FakeFactory.configurations = copy.deepcopy(configs)
        ledger.start(bundle)
        self.trace(['L','R'])
        if self.mode == 'discover_reject':
            raise job.Reject('CONFIGURATIONS_EXHAUSTED', configurations=len(configs))
        if self.mode == 'discover_budget':
            raise job.BudgetExceeded('discovery test limit')
        if self.mode == 'unexpected':
            raise RuntimeError('integrity-failure')
        candidate = {'histories': {'H_A': ['F'], 'H_B': ['L'], 'H_A_I': ['R']},
                     'continuations': {'C0': ['S'], 'C_A': ['F','S'], 'C_B': ['L','S']}}
        ledger.freeze(bundle, candidate)
        return candidate
    def replay_seeds(self, candidate):
        for seed in (1109,2209,3309):
            for h in candidate['histories']:
                for c in candidate['continuations']:
                    self.trace(candidate['histories'][h]+candidate['continuations'][c], seed)
                    if self.mode == 'cert_reject':
                        raise job.Reject('REAL_TRACE_INCOMPLETE')
                    if self.mode == 'cert_budget':
                        raise job.BudgetExceeded('certification test limit')
        return {'replays': 27, 'evaluations': 54}


class FamilyJobTests(unittest.TestCase):
    def setup(self, mode='pass', empty=False, budget=None, ledger=None):
        self.events = []
        FakeFactory.mode = mode
        cfg = {'bundle_id': 'MP5_00', 'task_a': 'task_A', 'task_b': 'task_B',
               'context': {'house_id': 'house', 'asset_config': {'scene': 'hash'}},
               'configurations': [{'index': 0, 'u_position': [1.123456789,2,3],
                                   'yaw_bin': 12, 'public_tail': 'LRLRLRLR'}]}
        compiler = types.SimpleNamespace(eligible={k: [i+1] for i,k in enumerate(['anchor_A','anchor_B','terminal','irrelevant'])})
        if empty:
            compiler.eligible['anchor_A'] = []
        budget = budget or BudgetLedger(clock=lambda: 1.0)
        ledger = ledger or FreezeLedger()
        return cfg, object(), compiler, budget, ledger, lambda kind,value: self.events.append((kind,value))

    def run_case(self, *a, **kw):
        args = self.setup(*a, **kw)
        with patch.object(job, 'FamilyFactory', FakeFactory):
            result = job.run_bundle(*args)
        return result, args

    def test_success_retains_only_27_certification_traces(self):
        result, args = self.run_case()
        self.assertEqual(result['status'], 'certified')
        self.assertEqual(len(result['certification_traces']), 27)
        self.assertEqual(result['counters']['trace_returns'], 28)
        self.assertEqual(result['counters']['confirmed_actions'], 47)
        self.assertEqual(result['counters']['executed_stops'], 27)
        self.assertEqual(args[4].records['MP5_00']['status'], 'certified')
        self.assertFalse(result['scientific_pass'])
        self.assertFalse(result['training_admission'])
        self.assertIsNone(args[3].snapshot()['active'])

    def test_coordinate_order_unchanged_and_source_not_mutated(self):
        args = self.setup()
        original = copy.deepcopy(args[0])
        with patch.object(job, 'FamilyFactory', FakeFactory):
            job.run_bundle(*args)
        self.assertEqual(FakeFactory.configurations, original['configurations'])
        self.assertEqual(args[0], original)

    def test_metadata_empty_no_factory_no_action(self):
        args = self.setup(empty=True)
        with patch.object(job, 'FamilyFactory', side_effect=AssertionError('NO_FACTORY')):
            result = job.run_bundle(*args)
        self.assertEqual(result['status'], 'metadata_ineligible')
        self.assertEqual(result['counters']['confirmed_actions'], 0)
        self.assertEqual(result['certification_traces'], [])

    def test_discovery_rejection_retains_trace_counts(self):
        result, args = self.run_case('discover_reject')
        self.assertEqual(result['status'], 'discovery_rejected')
        self.assertEqual(result['counters']['complete_traces'], 1)
        self.assertEqual(args[4].records['MP5_00']['status'], 'discovery_rejected')

    def test_discovery_budget_closes_without_reset(self):
        result, args = self.run_case('discover_budget')
        self.assertEqual(result['status'], 'resource_censored')
        self.assertEqual(args[3].snapshot()['total_reserved_actions'], 2)
        self.assertIsNone(args[3].snapshot()['active'])
        self.assertEqual(args[4].records['MP5_00']['status'], 'resource_censored')

    def test_certification_failure_cannot_replace_candidate(self):
        result, args = self.run_case('cert_reject')
        self.assertEqual(result['status'], 'certification_rejected')
        self.assertEqual(len(result['certification_traces']), 1)
        self.assertEqual(args[4].records['MP5_00']['status'], 'certification_rejected')
        with patch.object(job, 'FamilyFactory', FakeFactory), self.assertRaisesRegex(ValueError, 'ALREADY_ATTEMPTED'):
            job.run_bundle(*args)

    def test_cert_budget_is_resource_not_physical_negative(self):
        result, args = self.run_case('cert_budget')
        self.assertEqual(result['status'], 'resource_censored')
        self.assertEqual(result['details']['phase'], 'certification')
        self.assertTrue(args[4].records['MP5_00']['events'][-1]['reason'].startswith('RESOURCE_CENSORED:'))

    def test_same_batch_budget_not_reinitialized(self):
        result, args = self.run_case()
        first = args[3].snapshot()['total_reserved_actions']
        args[0]['bundle_id'] = 'MP5_01'
        with patch.object(job, 'FamilyFactory', FakeFactory):
            second = job.run_bundle(*args)
        self.assertEqual(second['budget_snapshot']['total_reserved_actions'], first*2)

    def test_integrity_failure_propagates_and_phase_closes(self):
        args = self.setup('unexpected')
        with patch.object(job, 'FamilyFactory', FakeFactory), self.assertRaisesRegex(RuntimeError, 'integrity-failure'):
            job.run_bundle(*args)
        self.assertIsNone(args[3].snapshot()['active'])
        self.assertEqual(args[4].records['MP5_00']['status'], 'discovering')

    def test_reject_already_active_batch_without_closing_it(self):
        args = self.setup()
        args[3].start_bundle('OTHER', 'discovery')
        with self.assertRaisesRegex(ValueError, 'ALREADY_ACTIVE'):
            job.run_bundle(*args)
        self.assertEqual(args[3].snapshot()['active'], ['OTHER','discovery'])

    def test_persistence_failure_not_swallowed(self):
        args = list(self.setup())
        def broken(kind,value):
            raise OSError('journal write failed')
        args[-1] = broken
        with patch.object(job, 'FamilyFactory', FakeFactory), self.assertRaises(OSError):
            job.run_bundle(*args)

    def test_true_timeout_phase_can_close(self):
        clock = [0.0]
        args = self.setup(budget=BudgetLedger(clock=lambda: clock[0]))
        class Timed(FakeFactory):
            def discover(self, configs, ledger, bundle):
                ledger.start(bundle)
                clock[0] = 721.0
                self.budget.check_time()
        with patch.object(job, 'FamilyFactory', Timed):
            result = job.run_bundle(*args)
        self.assertEqual(result['status'], 'resource_censored')
        self.assertEqual(args[3].snapshot()['last_clock'], 721.0)
        self.assertIsNone(args[3].snapshot()['active'])


if __name__ == '__main__':
    unittest.main()
