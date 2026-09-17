"""CPU-only adapter contract tests, never import actual Habitat or numpy."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runtime_habitat_backend_tested', HERE/'habitat_backend.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
ROLE = {'mpcat40': 'chair', 'room': 'dining room', 'raw_match': {'mode': 'token', 'value': 'chair'}}


class V(list):
    def tolist(self):
        return list(self)


class FakeArray:
    def __init__(self, kind):
        self.kind = kind
        self.shape = (224, 224, 3) if kind == 'rgb' else (224, 224)
        self.dtype = 'uint8' if kind == 'rgb' else 'uint32'
    def __getitem__(self, index):
        return self
    def tobytes(self):
        return self.kind.encode()


class BackendTests(unittest.TestCase):
    def bare(self):
        obj = object.__new__(b.HabitatBackend)
        obj._closed, obj._obs = False, {'collided': False}
        obj.counts = {'primitive_actions': 0, 'observation_records': 0}
        return obj

    def test_import_has_no_optional_dependencies(self):
        tree = ast.parse((HERE/'habitat_backend.py').read_text())
        names = {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
        self.assertFalse(names & {'numpy', 'habitat_sim', 'quaternion'})

    def test_permission_denial_precedes_import(self):
        for permit in (None, {}, {'runtime_allowed': False}, {'runtime_allowed': 1}):
            with patch.object(b.importlib, 'import_module', side_effect=AssertionError('IMPORT_NOT_ALLOWED')):
                with self.assertRaises(PermissionError):
                    b.Backend('/not/read/scene.glb', 0, {}, object(), permit)

    def test_permit_requires_exact_gpu_before_import(self):
        with patch.object(b.importlib, 'import_module', side_effect=AssertionError('IMPORT_NOT_ALLOWED')):
            with self.assertRaises(ValueError):
                b.Backend('/not/read/scene.glb', True, {}, object(), {'runtime_allowed': True})

    def test_permit_requires_absolute_path(self):
        with self.assertRaises(PermissionError):
            b.validate_permit('scene.glb', 0, {'runtime_allowed': True})

    def test_role_copy_and_no_letter_mapping(self):
        source = {'arbitrary_slot': copy.deepcopy(ROLE)}
        result = b.validate_roles(source)
        source['arbitrary_slot']['room'] = 'wrong'
        self.assertEqual(result['arbitrary_slot']['room'], 'dining room')
        record = {'mpcat40': 'chair', 'room': 'dining room', 'raw': 'arm chair#2'}
        self.assertTrue(b.role_matches(record, result['arbitrary_slot']))

    def test_token_word_boundary_and_category_room(self):
        for field, value in [('raw', 'wheelchair'), ('mpcat40', 'sofa'), ('room', 'living room')]:
            r = {'mpcat40': 'chair', 'room': 'dining room', 'raw': 'chair'}
            r[field] = value
            self.assertFalse(b.role_matches(r, ROLE))

    def test_exact_tv_not_monitor_alias(self):
        r = {'mpcat40': 'tv_monitor', 'room': 'living room', 'raw_match': {'mode': 'exact', 'value': 'tv'}}
        self.assertTrue(b.role_matches({'mpcat40': 'tv_monitor', 'room': 'living room', 'raw': 'TV'}, r))
        self.assertFalse(b.role_matches({'mpcat40': 'tv_monitor', 'room': 'living room', 'raw': 'monitor'}, r))

    def test_reject_implicit_roles_and_regex_configuration(self):
        for value in ({'A': ('chair', 'dining room')}, {'A': dict(ROLE, raw_match={'mode': 'regex', 'value': '.*'})}):
            with self.assertRaises(ValueError):
                b.validate_roles(value)

    def test_heading_y_up_left_turn(self):
        self.assertEqual([b.heading(0, -1), b.heading(-1, 0), b.heading(0, 1), b.heading(1, 0)], [0, 6, 12, 18])
        self.assertEqual(b.turns(-1), ['R'])
        self.assertEqual(b.turns(24), [])

    def test_straight_geometry(self):
        self.assertEqual(b.geometric_actions([[0,0,0], [0,0,-1]], 0, [0,0,-2]), ['F']*4+['L','R'])

    def test_geometry_no_physical_validity_claim(self):
        # A vertical waypoint needs zero ideal horizontal moves, NOT proof of travel.
        self.assertEqual(b.geometric_actions([[0,0,0], [0,2,0]], 0, [0,2,-1]), ['L','R'])

    def test_geometry_budget_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, 'CAP'):
            b.geometric_actions([[0,0,0], [0,0,-1]], 0, [0,0,-2], max_actions=5)
        with self.assertRaises(ValueError):
            b.geometric_actions([[0,0,float('nan')]], 0, [0,0,0])

    def test_routes_source_forbids_hidden_agent_control(self):
        tree = ast.parse((HERE/'habitat_backend.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        route = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'routes')
        attrs = {n.attr for n in ast.walk(route) if isinstance(n, ast.Attribute)}
        self.assertFalse(attrs & {'step', 'reset', 'initialize_agent', 'set_state', 'get_sensor_observations', 'GreedyGeodesicFollower'})

    def test_routes_runtime_mock_has_no_step_capability(self):
        obj = self.bare()
        obj.eligible, obj.objects = {'any': [5]}, {5: {'center': [0,0,-2]}}
        obj.route_diagnostics = []
        class Pathfinder:
            def snap_point(self, point):
                return V([0,0,-1])
            def find_path(self, path):
                path.points = [V(path.requested_start), V(path.requested_end)]
                return True
        obj.sim = types.SimpleNamespace(pathfinder=Pathfinder())
        obj.np = types.SimpleNamespace(float32='f32', asarray=lambda x, **k: V(x),
                                       isfinite=lambda x: types.SimpleNamespace(all=lambda: True))
        obj.hs = types.SimpleNamespace(ShortestPath=types.SimpleNamespace)
        self.assertEqual(list(obj.routes([0,0,0], 0, 'any')), [['F']*4+['L','R']])
        self.assertEqual(len(obj.route_diagnostics), 16)
        self.assertEqual(obj.counts['primitive_actions'], 0)

    def test_step_is_one_actual_call_and_stop_rejected(self):
        obj, calls = self.bare(), []
        obj.sim = types.SimpleNamespace(step=lambda a: (calls.append(a) or {'collided': True}))
        self.assertTrue(obj.step('F'))
        self.assertEqual(calls, ['move_forward'])
        self.assertEqual(obj.counts['primitive_actions'], 1)
        with self.assertRaises(ValueError):
            obj.step('S')
        self.assertEqual(len(calls), 1)

    def test_missing_collision_signal_is_error(self):
        obj = self.bare()
        obj.sim = types.SimpleNamespace(step=lambda a: {})
        with self.assertRaisesRegex(ValueError, 'COLLISION_SIGNAL'):
            obj.step('L')

    def observe_backend(self, store):
        obj = self.bare()
        obj._obs = {'rgb': FakeArray('rgb'), 'semantic': FakeArray('semantic')}
        obj.objects = {1: {}}
        obj.np = types.SimpleNamespace(uint8='uint8', uint32='uint32', ascontiguousarray=lambda x: x,
                                       unique=lambda x, **k: ([0, 1, 987, 65535], [1, 2, 3, 4]))
        obj.content_store = store
        obj._pose = lambda: {'position': [0,0,0], 'rotation': [1,0,0,0], 'sensors': {}}
        return obj

    def test_store_dict_hash_and_unknown_masks(self):
        store = types.SimpleNamespace(put_array=lambda a, k: {'pixel_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'path': '/not/policy'})
        record = self.observe_backend(store).observe()
        self.assertEqual(record['unknown_mask_ids'], [987])
        self.assertFalse(record['evidence_complete'])
        self.assertNotIn('path', record)
        self.assertEqual(record['pixels']['1'], 2)

    def test_store_wrong_hash_rejected(self):
        store = types.SimpleNamespace(put_array=lambda a,k: {'pixel_sha256': '0'*64})
        with self.assertRaisesRegex(ValueError, 'PIXEL_HASH_MISMATCH'):
            self.observe_backend(store).observe()

    def test_close_once_and_operations_after_close_fail(self):
        obj, calls = self.bare(), []
        obj.sim = types.SimpleNamespace(close=lambda: calls.append(1))
        obj.close(); obj.close()
        self.assertEqual(calls, [1])
        with self.assertRaisesRegex(RuntimeError, 'CLOSED'):
            obj.step('F')

    def test_reset_preserves_engine_call_order_and_yaw(self):
        obj, calls = self.bare(), []
        obj.counts.update(trace_initializations=0, explicit_resets=0)
        obj.hs = types.SimpleNamespace(AgentState=types.SimpleNamespace)
        obj.np = types.SimpleNamespace(float32='f32', asarray=lambda x, **k: V(x),
                                       quaternion=lambda *x: list(x))
        def initialize(index, state):
            calls.append(('initialize', index, state.position, state.rotation))
        obj.sim = types.SimpleNamespace(seed=lambda seed: calls.append(('seed', seed)),
                                        initialize_agent=initialize,
                                        reset=lambda: (calls.append(('reset',)) or {'rgb': 'real-reset-return'}))
        obj.reset([1,2,3], 6, 1109)
        self.assertEqual([x[0] for x in calls], ['seed', 'initialize', 'reset'])
        self.assertAlmostEqual(calls[1][3][0], 2**-.5)
        self.assertAlmostEqual(calls[1][3][2], 2**-.5)
        self.assertEqual(obj._obs, {'rgb': 'real-reset-return'})
        self.assertEqual(obj.counts['explicit_resets'], 1)

    def test_reconstruct_explicit_sensor_inference_not_hidden_steps(self):
        obj, calls = self.bare(), []
        obj.counts['explicit_reconstructions'] = 0
        obj.hs = types.SimpleNamespace(AgentState=types.SimpleNamespace)
        obj.np = types.SimpleNamespace(float32='f32', asarray=lambda x, **k: V(x), quaternion=lambda *x: list(x))
        agent = types.SimpleNamespace(set_state=lambda state, **kw: calls.append(('set_state', state, kw)))
        obj.sim = types.SimpleNamespace(get_agent=lambda index: agent,
                                        get_sensor_observations=lambda: (calls.append(('render',)) or {}))
        obj.reconstruct({'position': [0,0,0], 'rotation': [1,0,0,0], 'sensors': {'rgb': {}, 'semantic': {}}})
        self.assertEqual([x[0] for x in calls], ['set_state', 'render'])
        self.assertEqual(calls[0][2], {'reset_sensors': True, 'infer_sensor_states': True})
        self.assertEqual(obj.counts['primitive_actions'], 0)
        self.assertEqual(obj.counts['explicit_reconstructions'], 1)


if __name__ == '__main__':
    unittest.main()
