import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact
from factory import inverse, FamilyFactory

def local_module(name):
    spec = importlib.util.spec_from_file_location('compact_test_'+name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pose(x=0):
    basic = {'position': [x, 0, 0], 'rotation': [1, 0, 0, 0]}
    return dict(basic, sensors={key: copy.deepcopy(basic) for key in ('rgb', 'semantic')})


class Tests(unittest.TestCase):
    def factory(self, closed=True, pattern=True):
        value = object.__new__(compact.CompactLoopFactory)
        outgoing = ['F'] * 10 + ['L', 'R']
        calls, events = [], []
        value.routes = lambda *args: iter([outgoing])
        value.pattern = lambda *args: pattern
        def replay(position, yaw, actions):
            calls.append(list(actions))
            return {'observations': [{'pose': pose()}, {'pose': pose(0 if closed else .001)}], 'trace_hash': '0'*64}
        value.runner = SimpleNamespace(run=replay)
        value.emit = lambda kind, data: events.append((kind, data))
        return value, outgoing, calls, events

    def test_only_actual_compact_sequence_replayed(self):
        factory, outgoing, calls, events = self.factory()
        got = factory.loop([0, 0, 0], 0, 'anchor_A', ['anchor_B'])
        expanded, compact_return = inverse(outgoing)
        self.assertEqual(got, outgoing + compact_return)
        self.assertEqual(calls, [outgoing, got])
        self.assertNotIn(outgoing + expanded, calls)
        self.assertFalse(events[-1][1]['unused_expanded_trajectory_executed'])

    def test_actual_return_error_rejected(self):
        factory, _, _, _ = self.factory(closed=False)
        with self.assertRaisesRegex(compact.Reject, 'NO_VALID_LOOP'):
            factory.loop([0, 0, 0], 0, 'anchor_A', [])

    def test_event_failure_not_accepted(self):
        factory, _, calls, _ = self.factory(pattern=False)
        with self.assertRaisesRegex(compact.Reject, 'NO_VALID_LOOP'):
            factory.loop([0, 0, 0], 0, 'anchor_A', [])
        self.assertEqual(len(calls), 1)

    def test_sensor_error_rejected_even_if_agent_closes(self):
        first, last = pose(), pose()
        last['sensors']['rgb']['position'][0] = .001
        self.assertFalse(compact.closure_report(first, last)['pass'])

    def test_missing_sensor_rejected(self):
        with self.assertRaisesRegex(compact.Reject, 'LOOP_SENSOR_SET'):
            compact.closure_report(pose(), {'position': [0,0,0], 'rotation':[1,0,0,0]})

    def test_downstream_validators_not_overridden(self):
        for name in ('histories', 'validate_matrix', 'replay_seeds'):
            self.assertIs(getattr(compact.CompactLoopFactory, name), getattr(FamilyFactory, name))

    def test_non_neutral_tail_fails_before_loop_discovery(self):
        factory, _, calls, _ = self.factory(pattern=False)
        factory.a, factory.b, factory.end = 'A', 'B', 'T'
        with self.assertRaisesRegex(compact.Reject, 'INITIAL_PUBLIC_TAIL_NOT_NEUTRAL'):
            factory.construct({'u_position':[0,0,0], 'yaw_bin':0, 'public_tail':'LRLRLRLR'})
        self.assertEqual(calls, [list('LRLRLRLR')])

    def test_source_positions_require_all_roles(self):
        backend = SimpleNamespace(eligible={'A': [1], 'B': [2]}, snap_position=lambda p:p)
        def targets(b, p, role):
            if p[0] == 1 and role == 'B': return []
            return [{'distance': p[0]+1}]
        with patch.object(compact, 'candidate_targets', targets):
            selected, report = compact.rank_reachable_positions(backend, [[1,0,0],[0,0,0]])
        self.assertEqual([row['position'] for row in selected], [[0,0,0]])
        self.assertEqual(report['all_roles_reachable_positions'], 1)

    def test_not_snapping_invalid_source_start(self):
        backend = SimpleNamespace(eligible={'A': [1]}, snap_position=lambda p:[0,0,0])
        with patch.object(compact, 'candidate_targets', return_value=[{'distance':1}]):
            selected, _ = compact.rank_reachable_positions(backend, [[.01,0,0]])
        self.assertEqual(selected, [])

    def test_worker_rewrite_parses(self):
        worker = local_module('worker')
        source = worker.adapted_source()
        self.assertEqual(source.count('rank_reachable_positions(backend,'), 1)
        self.assertNotIn('positions=sorted(positions,key=', source)

    def test_gpu_graphics_and_conservative_cap(self):
        run = local_module('run')
        raw = '<nvidia_smi_log><gpu><uuid>'+run.UUID+'</uuid><fb_memory_usage><used>1000 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes><process_info><pid>10</pid><used_memory>500 MiB</used_memory><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
        snapshot = run.parse_gpu(raw)
        self.assertEqual(snapshot['processes'][10]['type'], 'G')
        self.assertEqual(run.check_gpu(snapshot,10), 1000)
        snapshot['memory_mib'] = 4096
        with self.assertRaisesRegex(AssertionError, 'OWN_GPU_MEMORY'):
            run.check_gpu(snapshot,10)

    def test_external_task_never_treated_as_own(self):
        run = local_module('run')
        with self.assertRaisesRegex(AssertionError, 'EXTERNAL_RESOURCE_LOAD'):
            run.check_gpu({'processes':{99:{'mib':1000}}, 'memory_mib':1100}, 10)

if __name__ == '__main__': unittest.main()
