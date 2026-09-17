"""CPU fixtures and sealed-log playback. Never simulated new physical data."""
import copy
import json
from pathlib import Path
import unittest

from compiler import Compiler, digest
from factory import FamilyFactory, TraceRunner, Reject, inverse, compress, pose_distance, yaw_bin
from planning import BudgetLedger, BudgetExceeded, FreezeLedger, cache_key

LINE = Path(__file__).resolve().parents[2]
DATA = LINE/'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'
CONTEXT = {'house_id': '17DRP5sb8fy', 'asset_config': {'source': 'sealed_CPU_log_fixture_not_runtime_asset_authorization'}}


def fixture_compiler():
    inventory = json.loads((DATA/'TASK_ROLE_INVENTORY.json').read_text())
    mapping = {'D': 'anchor_A', 'K': 'anchor_B', 'B': 'terminal', 'L': 'irrelevant'}
    roles = {mapping[k]: v for k, v in inventory['kinds'].items()}
    eligible = {mapping[k]: v for k, v in inventory['eligible'].items()}
    tasks = {'A': {'anchor': 'anchor_A', 'terminal': 'terminal', 'instruction': 'anchor A then terminal'},
             'B': {'anchor': 'anchor_B', 'terminal': 'terminal', 'instruction': 'anchor B then terminal'}}
    return Compiler(roles, tasks, eligible)


def candidate_fixture():
    old = json.loads((LINE/'reviews/Q35N_G1R_TASK_INSTANCE_V3/FROZEN_CANDIDATE.json').read_text())
    hmap, cmap = {'H_T': 'H_A', 'H_K': 'H_B', 'H_T_I': 'H_A_I'}, {'C0': 'C0', 'C_T': 'C_A', 'C_K': 'C_B'}
    compiler = fixture_compiler()
    key = cache_key(CONTEXT['house_id'], CONTEXT['asset_config'], {k: list(v) for k, v in compiler.roles.items()},
                    extra={'eligible': dict(compiler.eligible), 'tasks': {k: dict(v) for k, v in compiler.tasks.items()}})
    candidate = {'context': copy.deepcopy(CONTEXT), 'context_key': key,
            'position': old['u']['position'], 'yaw_bin': old['yaw_bin'], 'public_tail': old['public_tail'],
            'histories': {hmap[k]: v for k, v in old['histories'].items()},
            'continuations': {cmap[k]: v for k, v in old['continuations'].items()},
            'numerical_join': old['numerical_join']}
    candidate['candidate_hash'] = digest(candidate)
    return candidate


class RecordedBackend:
    """An ordered file reader, not an actual renderer or physical simulator."""
    def __init__(self, traces):
        self.traces = iter(traces)
        self.steps, self.joins = 0, 0

    def reset(self, position, yaw, seed):
        self.current = next(self.traces)
        assert self.current['initial_position'] == position
        assert self.current['initial_yaw_bin'] == yaw and self.current['seed'] == seed
        self.t, self.normalized = 0, False

    def observe(self):
        for event in self.current.get('normalization_events', []):
            if self.t == event['step'] and not self.normalized:
                return copy.deepcopy(event['raw_record'])
        return copy.deepcopy(self.current['observations'][self.t])

    def step(self, action):
        assert self.current['actions'][self.t] == action
        self.t += 1
        self.steps += 1
        self.normalized = False
        return False

    def reconstruct(self, target):
        assert digest(target) == digest(self.current['observations'][self.t]['pose'])
        self.normalized = True
        self.joins += 1


def traces_fixture():
    return [json.loads((DATA/'physical_traces'/f'{seed}_{h}_{c}.json').read_text())
            for seed in (1109, 2209, 3309) for h in ('H_T', 'H_K', 'H_T_I') for c in ('C0', 'C_T', 'C_K')]


def budget(limit=200000):
    b = BudgetLedger({'total_actions': limit, 'total_seconds': 6000, 'discovery_actions': limit,
                      'discovery_seconds': 720, 'certification_actions': limit, 'certification_seconds': 480}, clock=lambda: 0.)
    b.start_bundle('cpu_fixture')
    return b


class FactoryTests(unittest.TestCase):
    def test_01_all_27_sealed_log_playbacks(self):
        backend, ledger = RecordedBackend(traces_fixture()), budget()
        factory = FamilyFactory(backend, fixture_compiler(), ledger, lambda *args: None, 'A', 'B', context=CONTEXT)
        result = factory.replay_seeds(candidate_fixture())
        self.assertEqual((result['replays'], result['evaluations']), (27, 54))
        self.assertEqual(backend.joins, 27)
        self.assertEqual(backend.steps, 8631)
        self.assertEqual(ledger.snapshot()['total_reserved_actions'], backend.steps)
        self.assertEqual(factory.runner.counts['confirmed_action_returns'], backend.steps)
        self.assertEqual(factory.runner.counts['executed_stops'], 27)
        self.assertFalse(result['training_admission'])
        for seed in result['seeds']:
            self.assertEqual(sum(r['outcome'] == 'pass' for r in seed['rows']), 12)

    def test_02_budget_precedes_backend_step(self):
        backend = RecordedBackend(traces_fixture())
        runner = TraceRunner(backend, fixture_compiler(), budget(1), lambda *args: None)
        candidate = candidate_fixture()
        with self.assertRaises(BudgetExceeded):
            runner.run(candidate['position'], candidate['yaw_bin'], candidate['histories']['H_A'])
        self.assertEqual(backend.steps, 1)

    def test_03_invalid_stop_rejected_before_reset(self):
        runner = TraceRunner(object(), fixture_compiler(), budget(), lambda *args: None)
        with self.assertRaisesRegex(Reject, 'ACTION_SEQUENCE'):
            runner.run([0, 0, 0], 0, ['S', 'F'])

    def test_04_inverse_and_compression(self):
        self.assertEqual(compress(['L', 'R', 'F', 'R', 'L']), ['F'])
        self.assertEqual(inverse(['L', 'R'])[1], [])
        self.assertEqual(inverse(['F'])[1], ['L']*12+['F']+['L']*12)
        with self.assertRaises(Reject):
            inverse(['S'])

    def test_05_quaternion_sign_equivalence(self):
        a = {'position': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
        b = {'position': [0, 0, 0], 'rotation': [-1, 0, 0, 0]}
        self.assertEqual(pose_distance(a, b), (0., 0.))
        self.assertEqual(yaw_bin(a), 0)
        with self.assertRaises(Reject):
            pose_distance(a, dict(b, rotation=[0, 0, 0, 0]))

    def test_06_wrong_join_target_rejected(self):
        candidate = candidate_fixture()
        candidate['numerical_join']['target_pose']['position'][0] += 1e-6
        runner = TraceRunner(RecordedBackend(traces_fixture()), fixture_compiler(), budget(), lambda *args: None)
        with self.assertRaisesRegex(Reject, 'JOIN_TARGET_NOT_INITIAL_STATE'):
            runner.run(candidate['position'], candidate['yaw_bin'], candidate['histories']['H_A']+candidate['continuations']['C0'], join=candidate['numerical_join'])

    def test_07_large_raw_join_rejected(self):
        traces, candidate = traces_fixture(), candidate_fixture()
        traces[0]['normalization_events'][0]['raw_record']['pose']['position'][0] += 0.01
        runner = TraceRunner(RecordedBackend(traces), fixture_compiler(), budget(), lambda *args: None)
        with self.assertRaisesRegex(Reject, 'JOIN_CORRECTION_TOO_LARGE'):
            runner.run(candidate['position'], candidate['yaw_bin'], candidate['histories']['H_A']+candidate['continuations']['C0'], join=candidate['numerical_join'])

    def test_08_changed_boundary_event_rejected(self):
        traces, candidate = traces_fixture(), candidate_fixture()
        event = traces[0]['normalization_events'][0]
        t = event['step']
        idx = fixture_compiler().eligible['anchor_A'][0]
        traces[0]['observations'][t-1]['pixels'][str(idx)] = 300
        event['raw_record']['pixels'][str(idx)] = 300
        traces[0]['observations'][t]['pixels'][str(idx)] = 0
        runner = TraceRunner(RecordedBackend(traces), fixture_compiler(), budget(), lambda *args: None)
        with self.assertRaisesRegex(Reject, 'JOIN_EVENTS_CHANGED'):
            runner.run(candidate['position'], candidate['yaw_bin'], candidate['histories']['H_A']+candidate['continuations']['C0'], join=candidate['numerical_join'])

    def test_09_changed_short_window_rejected(self):
        traces, candidate = traces_fixture(), candidate_fixture()
        t = len(candidate['histories']['H_A'])
        traces[1]['observations'][t]['rgb_hash'] = '1'*64
        factory = FamilyFactory(RecordedBackend(traces), fixture_compiler(), budget(), lambda *args: None, 'A', 'B', context=CONTEXT)
        with self.assertRaisesRegex(Reject, 'MERGE_SHORT_WINDOW_MISMATCH'):
            factory.validate_matrix(candidate)

    def test_10_first_preflight_freezes_and_stops(self):
        # State-machine unit fixture; not a semantic or physical family.
        factory = FamilyFactory(object(), fixture_compiler(), budget(), lambda *args: None, 'A', 'B', context=CONTEXT)
        calls = []
        def construct(config):
            calls.append(config)
            if config == 0:
                raise Reject('fixture_reject')
            return {'fixture': config}
        factory.construct = construct
        frozen = FreezeLedger()
        self.assertEqual(factory.discover([0, 1, 2], frozen, 'fixture'), {'fixture': 1})
        self.assertEqual(calls, [0, 1])

    def test_11_bad_matrix_shape_rejected_before_backend(self):
        factory = FamilyFactory(object(), fixture_compiler(), budget(), lambda *args: None, 'A', 'B', context=CONTEXT)
        candidate = candidate_fixture()
        del candidate['histories']['H_B']
        with self.assertRaisesRegex(Reject, 'FAMILY_MATRIX'):
            factory.validate_matrix(candidate)

    def test_12_literal_scope_no_runtime_imports(self):
        import ast
        source = Path(__file__).with_name('factory.py').read_text()
        tree = ast.parse(source)
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(modules & {'habitat_sim', 'torch', 'numpy', 'transformers'})

    def test_13_wrong_house_context_rejected(self):
        factory = FamilyFactory(object(), fixture_compiler(), budget(), lambda *args: None, 'A', 'B', context=CONTEXT)
        candidate = candidate_fixture()
        candidate['context']['house_id'] = 'different_house'
        with self.assertRaisesRegex(Reject, 'CANDIDATE_CONTEXT_MISMATCH'):
            factory.validate_matrix(candidate)

    def test_14_pixel_key_collision_rejected(self):
        traces = traces_fixture()
        traces[0]['observations'][0]['pixels'] = {1: 300, '1': 0}
        runner = TraceRunner(RecordedBackend(traces), fixture_compiler(), budget(), lambda *args: None)
        candidate = candidate_fixture()
        with self.assertRaisesRegex(Reject, 'PIXEL_KEY_COLLISION'):
            runner.run(candidate['position'], candidate['yaw_bin'], [])

    def test_15_final_observation_timeout_rejected(self):
        now = [0.]
        ledger = BudgetLedger(clock=lambda: now[0])
        ledger.start_bundle('fixture')
        backend = RecordedBackend(traces_fixture())
        observe = backend.observe
        def delayed():
            result = observe()
            now[0] = 800.
            return result
        backend.observe = delayed
        runner = TraceRunner(backend, fixture_compiler(), ledger, lambda *args: None)
        candidate = candidate_fixture()
        with self.assertRaises(BudgetExceeded):
            runner.run(candidate['position'], candidate['yaw_bin'], [])

    def test_16_join_next_event_rejected(self):
        traces, candidate = traces_fixture(), candidate_fixture()
        event = traces[0]['normalization_events'][0]
        t, idx = event['step'], fixture_compiler().eligible['anchor_A'][0]
        traces[0]['observations'][t-1]['pixels'][str(idx)] = 0
        event['raw_record']['pixels'][str(idx)] = 0
        traces[0]['observations'][t]['pixels'][str(idx)] = 300
        traces[0]['observations'][t+1]['pixels'][str(idx)] = 300
        runner = TraceRunner(RecordedBackend(traces), fixture_compiler(), budget(), lambda *args: None)
        with self.assertRaisesRegex(Reject, 'JOIN_NEXT_EVENTS_CHANGED'):
            runner.run(candidate['position'], candidate['yaw_bin'], candidate['histories']['H_A']+candidate['continuations']['C0'], join=candidate['numerical_join'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
