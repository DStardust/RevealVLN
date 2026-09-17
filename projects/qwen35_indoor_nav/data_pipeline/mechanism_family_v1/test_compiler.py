"""Synthetic unit tests only; never a source of mechanism training labels."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


c = module('compiler'); v = module('schema_check')
SCHEMA = json.loads((HERE.parents[1] / 'reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1/DATA_SCHEMA_V2.json').read_text())
ELIGIBLE = {'D': [1, 2], 'K': [3], 'B': [4], 'L': [5]}


def fixture():
    obs = []
    for t, pixels in enumerate([{1: 300}, {1: 300}, {}, {4: 400}, {4: 400}]):
        obs.append({'step': t, 'pixels': pixels, 'evidence_complete': True,
                    'rgb_hash': c.digest(['rgb', t]), 'semantic_hash': c.digest(['sem', t])})
    return {'actions': list('LRFFS'), 'observations': obs, 'complete': True,
            'collisions': 0, 'trace_hash': c.digest('synthetic-unit-only')}


class CompilerTest(unittest.TestCase):
    def test_vocabulary_is_supported(self):
        v.vocabulary_check(SCHEMA)
        with self.assertRaises(ValueError): v.vocabulary_check({'unevaluatedProperties': False})

    def test_hash_json_roundtrip(self):
        value = {'pixels': {100: 3, 2: 4, 10: 5}}
        self.assertEqual(c.digest(value), c.digest(json.loads(json.dumps(value))))
        with self.assertRaises(ValueError): c.digest({1: 'a', '1': 'b'})

    def test_schema_does_not_accept_boolean_as_integer(self):
        with self.assertRaises(ValueError): v.validate(True, {'type': 'integer'})
        with self.assertRaises(ValueError): v.validate(True, {'const': 1})
        with self.assertRaises(ValueError): v.validate(True, {'enum': [0, 1]})

    def test_order_and_missing_evidence(self):
        tr = fixture()
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'pass')
        self.assertEqual(c.evaluate(tr, 'g_K_v2', ELIGIBLE), 'fail')
        tr['observations'][1]['evidence_complete'] = False
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'unknown')

    def test_same_instance_required(self):
        tr = fixture(); tr['observations'][1]['pixels'] = {2: 400}
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'fail')

    def test_threshold(self):
        tr = fixture(); tr['observations'][1]['pixels'] = {1: 255}
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'fail')
        tr['observations'][1]['pixels'] = {'1': 256}
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'pass')

    def test_simultaneous_first_anchor_and_bed_not_ready(self):
        tr = fixture()
        for o in tr['observations']: o['pixels'] = {}
        for o in tr['observations'][-2:]: o['pixels'] = {1: 300, 4: 300}
        self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'fail')
        self.assertEqual(c.m2(tr, 'g_D_v2', ELIGIBLE)[-1]['state'], 'WAIT_TERMINAL_WITNESS')

    def test_query_schema_and_storage_invariance(self):
        q = c.query_from_trace(fixture(), ELIGIBLE)
        v.validate(q, SCHEMA['$defs']['continuation_query'], SCHEMA)
        changed = copy.deepcopy(q)
        changed['action_trace_ref'] = c.ref('renamed')
        changed['rgb_content_refs'] = []
        self.assertEqual(c.canonical(c.semantic_query(q)), c.canonical(c.semantic_query(changed)))
        self.assertEqual(c.encode_query(q), c.encode_query(changed))
        changed['sequence'][0]['house_id'] = 'forbidden'
        with self.assertRaises(ValueError): v.validate(changed, SCHEMA['$defs']['continuation_query'], SCHEMA)

    def test_policy_future_exclusion(self):
        tr = fixture(); p = c.policy_at(tr, 'g_D_v2', 2, 'test')
        v.validate(p, SCHEMA)
        self.assertEqual([o['step'] for o in p['observations']], [1, 2])
        self.assertEqual([a['step'] for a in p['executed_actions']], [0, 1])
        changed = copy.deepcopy(tr)
        changed['actions'][2:] = list('RLS')
        changed['observations'][3]['rgb_hash'] = c.digest('altered-future')
        self.assertEqual(p, c.policy_at(changed, 'g_D_v2', 2, 'test'))

    def test_m2_is_causal(self):
        tr = fixture(); before = c.m2(tr, 'g_D_v2', ELIGIBLE)
        tr['observations'][-1]['evidence_complete'] = False
        after = c.m2(tr, 'g_D_v2', ELIGIBLE)
        self.assertEqual(before[:-1], after[:-1])
        self.assertEqual(after[-1]['loss_mask'], 0)

    def test_event_schema(self):
        for taskkind in ('D', 'K', 'B'):
            cert = c.event_certificate(fixture(), taskkind, ELIGIBLE, 'complete_interval' if taskkind != 'B' else 'decision_time')
            v.validate(cert, SCHEMA['$defs']['event_certificate'], SCHEMA)

    def test_trace_corruption_fails(self):
        for mutate in (lambda tr: tr['observations'].pop(1),
                       lambda tr: tr.update(collisions=1),
                       lambda tr: tr['actions'].__setitem__(1, 'S')):
            tr = fixture(); mutate(tr)
            self.assertEqual(c.evaluate(tr, 'g_D_v2', ELIGIBLE), 'unknown')

    def test_nonpass_mask_schema(self):
        constraint = SCHEMA['$defs']['supervision_only']['allOf'][1]
        with self.assertRaises(ValueError): v.validate({'outcome': 'fail', 'action_loss_mask': [1]}, constraint, SCHEMA)


if __name__ == '__main__': unittest.main()
