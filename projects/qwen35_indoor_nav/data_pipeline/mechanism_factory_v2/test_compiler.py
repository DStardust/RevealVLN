"""Synthetic edge tests plus read-only real 27 x 2 differential checks."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c = load(HERE / 'compiler.py', 'factory_compiler')


def configured():
    return c.Compiler({'anchor_A': ('tv_monitor', 'living room'),
                       'anchor_B': ('sink', 'kitchen'), 'terminal': ('bed', 'bedroom')},
                      {'a': {'anchor': 'anchor_A', 'terminal': 'terminal', 'instruction': 'TV then bed; stop'},
                       'b': {'anchor': 'anchor_B', 'terminal': 'terminal', 'instruction': 'sink then bed; stop'}},
                      {'anchor_A': [1, 2], 'anchor_B': [3], 'terminal': [4]})


def fixture():
    return {'actions': list('LRFFS'), 'observations': [
        {'step': t, 'pixels': pixels, 'evidence_complete': True,
         'rgb_hash': c.digest(['synthetic_rgb', t]), 'semantic_hash': c.digest(['synthetic_sem', t])}
        for t, pixels in enumerate([{1: 256}, {1: 256}, {}, {4: 300}, {4: 300}])],
        'complete': True, 'collisions': 0, 'trace_hash': c.digest('synthetic_unit_only')}


def semantic_timeline(query):
    """Compare ordered movement and per-step event sets across V2/V4 schemas."""
    out, observations = [], []
    for item in query['sequence']:
        semantic = {k: v for k, v in item.items() if k != 'order'}
        if item['kind'] == 'observe':
            observations.append(semantic)
        else:
            if observations:
                out.append(sorted(observations, key=c.canonical))
                observations = []
            out.append(semantic)
    if observations:
        out.append(sorted(observations, key=c.canonical))
    return out


class CompilerTests(unittest.TestCase):
    def test_threshold_and_same_instance(self):
        compiler, trace = configured(), fixture()
        self.assertEqual(compiler.evaluate(trace, 'a'), 'pass')
        for pixels in ({1: 255}, {2: 500}):
            trace['observations'][1]['pixels'] = pixels
            self.assertEqual(compiler.evaluate(trace, 'a'), 'fail')
        trace['observations'][1]['pixels'] = {'1': 256}
        self.assertEqual(compiler.evaluate(trace, 'a'), 'pass')

    def test_anchor_strictly_before_terminal(self):
        trace, compiler = fixture(), configured()
        for o in trace['observations']:
            o['pixels'] = {}
        for o in trace['observations'][-2:]:
            o['pixels'] = {1: 300, 4: 300}
        self.assertEqual(compiler.evaluate(trace, 'a'), 'fail')
        self.assertEqual(compiler.m2(trace, 'a')[-1]['state'], 'WAIT_TERMINAL_WITNESS')

    def test_unknown_evidence(self):
        compiler = configured()
        for mutation in (lambda x: x.update(complete=False), lambda x: x.update(collisions=1),
                         lambda x: x['observations'][1].update(evidence_complete=False),
                         lambda x: x['observations'][1].update(pixels={'1': -1}),
                         lambda x: x['observations'][1].update(pixels={'1': True})):
            trace = fixture()
            mutation(trace)
            self.assertEqual(compiler.evaluate(trace, 'a'), 'unknown')
            with self.assertRaises(ValueError):
                compiler.query_from_trace(trace)

    def test_illegal_stop_time_action(self):
        compiler = configured()
        for mutation in (lambda x: x['actions'].__setitem__(1, 'S'),
                         lambda x: x['actions'].__setitem__(1, 'X'),
                         lambda x: x['observations'][1].update(step=2),
                         lambda x: x['observations'][1].update(step=True),
                         lambda x: x['observations'].pop()):
            trace = fixture()
            mutation(trace)
            self.assertEqual(compiler.evaluate(trace, 'a'), 'unknown')

    def test_no_stop_is_known_failure(self):
        trace = fixture()
        trace['actions'].pop()
        self.assertTrue(c.complete(trace))
        self.assertEqual(configured().evaluate(trace, 'a'), 'fail')

    def test_config_isolation_and_immutable_nested_values(self):
        roles = {'x': ['tv_monitor', 'living room'], 'y': ['bed', 'bedroom']}
        tasks = {'a': {'anchor': 'x', 'terminal': 'y', 'instruction': 'same text'}}
        eligible = {'x': [1], 'y': [4]}
        first = c.Compiler(roles, tasks, eligible)
        eligible['x'][0] = 8
        roles['x'][0] = 'chair'
        tasks['a']['instruction'] = 'changed'
        second = c.Compiler(roles, tasks, eligible)
        self.assertEqual(first.evaluate(fixture(), 'a'), 'pass')
        self.assertEqual(second.evaluate(fixture(), 'a'), 'fail')
        self.assertEqual(first.tasks['a']['instruction'], 'same text')
        with self.assertRaises(TypeError):
            first.tasks['a']['anchor'] = 'y'

    def test_role_rename_query_and_label_invariance(self):
        old = configured()
        rename = {'anchor_A': 'zz', 'anchor_B': 'middle', 'terminal': 'aa'}
        new = c.Compiler({rename[k]: old.roles[k] for k in reversed(old.roles)},
                         {k: dict(v, anchor=rename[v['anchor']], terminal=rename[v['terminal']])
                          for k, v in old.tasks.items()},
                         {rename[k]: old.eligible[k] for k in old.eligible})
        trace = fixture()
        for o in trace['observations'][-2:]:
            o['pixels'][3] = 300
        self.assertEqual(old.query_from_trace(trace), new.query_from_trace(trace))
        self.assertEqual(old.encode_query(old.query_from_trace(trace)), new.encode_query(new.query_from_trace(trace)))
        self.assertEqual(old.evaluate(trace, 'a'), new.evaluate(trace, 'a'))

    def test_unknown_task_and_invalid_config(self):
        with self.assertRaises(KeyError):
            configured().evaluate(fixture(), 'missing')
        with self.assertRaises(ValueError):
            c.Compiler({'x': ('bed', 'bedroom')}, {'a': {'anchor': 'x', 'terminal': 'x', 'instruction': 'x'}}, {'x': [True]})
        with self.assertRaises(ValueError):
            c.Compiler({'x': ('bed', 'bedroom')}, {'a': {'anchor': 'x', 'terminal': 'x', 'instruction': 'x'}}, {'x': [1]}, sensor={})

    def test_m2_and_policy_future_causality(self):
        compiler, trace = configured(), fixture()
        state, policy = compiler.m2(trace, 'a'), compiler.policy_at(trace, 'a', 2, 'example')
        trace['observations'][-1].update(evidence_complete=False, rgb_hash='future_changed')
        trace['actions'][3] = 'X'
        self.assertEqual(state[:3], compiler.m2(trace, 'a')[:3])
        self.assertEqual(policy, compiler.policy_at(trace, 'a', 2, 'example'))
        self.assertEqual(compiler.m2(trace, 'a')[-1]['loss_mask'], 0)

    def test_policy_privileged_metadata_excluded(self):
        compiler, trace = configured(), fixture()
        original = compiler.policy_at(trace, 'a', 2, 'record_only')
        trace.update(house_id='secret', oracle_y='pass', query={'future': 'secret'})
        for o in trace['observations']:
            o.update(pixels={'987': 400}, semantic_hash='secret', pose={'secret': 1})
        self.assertEqual(original, compiler.policy_at(trace, 'a', 2, 'record_only'))
        self.assertNotIn('sample_id', c.policy_semantics(original))
        changed = dict(original, house_id='secret')
        self.assertEqual(c.policy_semantics(original), c.policy_semantics(changed))
        self.assertEqual([x['step'] for x in original['observations']], [1, 2])
        self.assertEqual([x['step'] for x in original['executed_actions']], [0, 1])

    def test_query_order_threshold_metadata_rejected(self):
        compiler = configured()
        original = compiler.query_from_trace(fixture())
        for mutation in (lambda q: q['sequence'][0].update(order=1),
                         lambda q: q['sequence'][0].update(house_id='secret'),
                         lambda q: q['sequence'][0].update(repeat=True),
                         lambda q: next(x for x in q['sequence'] if x['kind'] == 'observe').update(min_pixels=255),
                         lambda q: q['sequence'].append({'kind': 'act', 'action': 'STOP', 'order': len(q['sequence'])})):
            query = copy.deepcopy(original)
            mutation(query)
            with self.assertRaises(ValueError):
                compiler.encode_query(query)
        changed = dict(original, action_trace_ref='renamed', rgb_content_refs=[])
        self.assertEqual(compiler.encode_query(original), compiler.encode_query(changed))

    def test_action_dedup_stability_and_failure_mask(self):
        compiler, trace = configured(), fixture()
        keys, payloads = compiler.action_keys(trace, 'a', 'history', 2)
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys, compiler.action_keys(trace, 'a', 'history', 2)[0])
        self.assertEqual(payloads[-1]['target_action'], 'STOP')
        with self.assertRaises(ValueError):
            compiler.action_keys(trace, 'b', 'history', 2)

    def test_hash_and_slice(self):
        value = {'pixels': {100: 3, 2: 4, 10: 5}}
        self.assertEqual(c.digest(value), c.digest(json.loads(json.dumps(value))))
        with self.assertRaises(ValueError):
            c.digest({1: 'a', '1': 'b'})
        sliced = c.slice_continuation(fixture(), 2)
        self.assertTrue(c.complete(sliced))
        self.assertEqual([o['step'] for o in sliced['observations']], [0, 1, 2])

    def test_real_27_trace_54_labels_differential(self):
        source = LINE / 'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'
        inventory = json.loads((source / 'TASK_ROLE_INVENTORY.json').read_text())
        policies = [json.loads(line) for line in (source / 'POLICY_INPUT.jsonl').read_text().splitlines()]
        old = load(LINE / 'data_pipeline/mechanism_family_v1/compiler.py', 'legacy_compiler_readonly_differential')
        old.KINDS = {k: tuple(v) for k, v in inventory['kinds'].items()}
        old.TASK_ANCHOR = {'g_T_v3': 'D', 'g_K_v3': 'K'}
        old.TASKS = {p['sample_id'].rsplit('.', 1)[-1]: p['instruction'] for p in policies}
        old.TASK_REVISION = 'observable_tv_sink_then_stop.v3'
        tasks = {t: {'anchor': anchor, 'terminal': 'B', 'instruction': old.TASKS[t]}
                 for t, anchor in old.TASK_ANCHOR.items()}
        new = c.Compiler(inventory['kinds'], tasks, inventory['eligible'], task_revision=old.TASK_REVISION)
        paths = sorted((source / 'physical_traces').glob('*.json'))
        self.assertEqual(len(paths), 27)
        source_hashes = {str(p): c.digest(p.read_bytes()) for p in paths}
        outcomes, counted = {}, 0
        for path in paths:
            trace = json.loads(path.read_text())
            self.assertEqual(old.complete(trace), new.complete(trace))
            self.assertEqual(old.atoms(trace['observations'], inventory['eligible']), new.atoms(trace['observations']))
            for task in tasks:
                counted += 1
                outcome = new.evaluate(trace, task)
                self.assertEqual(old.evaluate(trace, task, inventory['eligible']), outcome)
                self.assertEqual(old.m2(trace, task, inventory['eligible']), new.m2(trace, task))
                # Version changed; deployment semantics and task revision did not.
                self.assertEqual(old.policy_semantics(old.policy_at(trace, task, 206, 'x')),
                                 c.policy_semantics(new.policy_at(trace, task, 206, 'x')))
                seed, rest = path.stem.split('_', 1)
                history, continuation = rest.rsplit('_C', 1)
                if seed == '1109':
                    outcomes[(task, history, 'C' + continuation)] = outcome
            query = new.query_from_trace(c.slice_continuation(trace, 206))
            old_query = old.query_from_trace(old.slice_continuation(trace, 206), inventory['eligible'])
            self.assertEqual(semantic_timeline(query), semantic_timeline(old_query))
            self.assertTrue(new.encode_query(query)['token_ids'])
        manifest = json.loads((source / 'FAMILY_MANIFEST.json').read_text())
        self.assertEqual(counted, 54)
        self.assertEqual(len(outcomes), 18)
        for cell in manifest['cross_cells']:
            self.assertEqual(outcomes[(cell['task_id'], cell['history_id'], cell['continuation_trace_id'])],
                             cell['paper_expected_outcome'])
        self.assertEqual(source_hashes, {str(p): c.digest(p.read_bytes()) for p in paths})


if __name__ == '__main__':
    unittest.main(verbosity=2)
