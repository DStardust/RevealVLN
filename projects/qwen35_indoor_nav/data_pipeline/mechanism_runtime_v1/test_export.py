"""CPU-only reexport of sealed physical logs, never new simulator replay."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SOURCE = LINE / 'reviews/Q35N_G1R_FAMILY_CERTIFICATION_V1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e = load('q35n_runtime_export_tests', HERE / 'exporter.py')
l = load('q35n_runtime_loader_tests', HERE / 'loader.py')
c = load('q35n_v4_compiler_export_tests', HERE.parent / 'mechanism_factory_v2/compiler.py')


def real_fixture():
    inventory = json.loads((SOURCE / 'TASK_ROLE_INVENTORY.json').read_text())
    policy = [json.loads(x) for x in (SOURCE / 'POLICY_INPUT.jsonl').read_text().splitlines()]
    instructions = {p['sample_id'].rsplit('.', 1)[-1]: p['instruction'] for p in policy}
    tasks = {t: {'anchor': role, 'terminal': 'B', 'instruction': instructions[t]}
             for t, role in (('g_T_v3', 'D'), ('g_K_v3', 'K'))}
    compiler = c.Compiler(inventory['kinds'], tasks, inventory['eligible'], task_revision='observable_tv_sink_then_stop.v3')
    old = json.loads((LINE / 'reviews/Q35N_G1R_TASK_INSTANCE_V3/FROZEN_CANDIDATE.json').read_text())
    candidate = {'histories': old['histories'], 'continuations': old['continuations'],
                 'context': {'house_id': '17DRP5sb8fy'}, 'source_kind': 'sealed_log_reexport_interface_only'}
    grids = [{(h, k): json.loads((SOURCE / 'physical_traces' / f'{seed}_{h}_{k}.json').read_text())
              for h in candidate['histories'] for k in candidate['continuations']} for seed in (1109, 2209, 3309)]
    return compiler, candidate, grids


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler, cls.candidate, cls.grids = real_fixture()
        cls.source_before = e.sha((SOURCE / 'SHA256SUMS').read_bytes())
        cls.temp = tempfile.TemporaryDirectory(dir=HERE, prefix='test_export_')
        cls.root = Path(cls.temp.name) / 'reexport'
        cls.result = e.export_family(cls.root, cls.compiler, cls.candidate, cls.grids[0], SOURCE / 'content',
                                     family_id='CPU_OLD_FAMILY_REEXPORT_NOT_NEW', repeated_traces=cls.grids[1:],
                                     provenance={'execution': 'CPU_only_readback_of_sealed_traces'})
        cls.loader = l.FamilyLoader(cls.root, cls.compiler)

    @classmethod
    def tearDownClass(cls):
        assert cls.source_before == e.sha((SOURCE / 'SHA256SUMS').read_bytes())
        cls.temp.cleanup()

    def test_01_real_counts(self):
        self.assertEqual(self.result['outcomes'], {'pass': 12, 'fail': 6})
        self.assertEqual((self.result['canonical_traces'], self.result['repeat_grids_checked']), (9, 2))
        self.assertEqual((self.result['prefixes'], self.result['prefix_decisions'], self.result['cells']), (6, 1242, 18))
        self.assertEqual(self.result['ce_unique_owners'], 1410)
        self.assertEqual(self.result['split'], 'interface_only')
        self.assertFalse(self.result['training_admission'])

    def test_02_all_causal_prefixes(self):
        count = 0
        for prefix_id in self.loader.prefix_index:
            records = self.loader.prefix_records(prefix_id)
            self.assertEqual(len(records), 207)
            count += len(records)
            payload = self.loader.policy_payload(records[-1])
            self.assertEqual(set(payload), {'instruction', 'task_type', 'rgb', 'executed_actions', 'memory_reset'})
            self.assertEqual([len(x) for x in payload['rgb']], [224*224*3]*2)
            self.assertEqual(len(payload['executed_actions']), 8)
            self.assertFalse(payload['memory_reset'])
        self.assertEqual(count, 1242)

    def test_03_query_order_and_independent_target(self):
        for cell in self.loader.cells:
            sup = self.loader.supervision(cell['cell_id'])
            self.assertEqual(sup['query'], self.compiler.encode_query(cell['query']))
            self.assertEqual([q['order'] for q in cell['query']['sequence']], list(range(len(cell['query']['sequence']))))
            trace = self.grids[0][cell['history_id'], cell['continuation_id']]
            self.assertEqual(cell['outcome'], self.compiler.evaluate(trace, cell['task_id']))
            self.assertEqual(cell['m2_program_state'], self.compiler.m2(trace, cell['task_id']))

    def test_04_ce_owners_are_pass_only(self):
        found = {}
        for cell in self.loader.cells:
            if cell['outcome'] != 'pass':
                self.assertEqual(cell['action_targets'], [])
            stream = self.loader.action_stream(cell['cell_id'])
            trace = self.grids[0][cell['history_id'], cell['continuation_id']]
            self.assertEqual(len(stream), len(trace['actions']))
            for index, (step, key, action, mask) in enumerate(zip(cell['action_steps'], cell['action_keys'], cell['action_targets'], cell['action_loss_mask'])):
                self.assertEqual(action, c.ACTIONS[trace['actions'][step]])
                self.assertEqual(stream[step]['causal_cutoff_step'], step)
                if mask:
                    self.assertNotIn(key, found)
                    found[key] = [cell['cell_id'], index]
        self.assertEqual(found, self.loader.owners['owners'])

    def test_05_policy_privileged_fields_rejected(self):
        record = self.loader.prefix_records(next(iter(self.loader.prefix_index)))[-1]
        for field in ('query', 'y', 'house_id', 'future', 'semantic_hash'):
            with self.assertRaisesRegex(ValueError, 'POLICY_FIELDS'):
                self.loader.policy_payload(dict(record, **{field: 'secret'}))

    def test_06_unknown_not_negative(self):
        cell = self.loader.cells[0]
        original = copy.deepcopy(cell)
        try:
            cell.update(outcome='unknown', y=None, bce_mask=0)
            self.assertIsNone(self.loader.supervision(cell['cell_id'])['y'])
            cell.update(y=0)
            with self.assertRaisesRegex(ValueError, 'UNKNOWN_IS_NOT_NEGATIVE'):
                self.loader.supervision(cell['cell_id'])
        finally:
            cell.clear()
            cell.update(original)

    def test_07_incomplete_rejected_before_write(self):
        grids = copy.deepcopy(self.grids[0])
        next(iter(grids.values()))['complete'] = False
        target = Path(self.temp.name) / 'invalid'
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_TRACE'):
            e.export_family(target, self.compiler, self.candidate, grids, SOURCE / 'content', family_id='invalid')
        self.assertFalse(target.exists())

    def test_08_scope_and_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
            e.export_family(self.root, self.compiler, self.candidate, self.grids[0], SOURCE / 'content', family_id='bad')
        with self.assertRaisesRegex(ValueError, 'PATH_ESCAPE'):
            l.safe_path(self.root, '../escape')
        with self.assertRaisesRegex(ValueError, 'RELATIVE_PATH'):
            l.safe_path(self.root, str(SOURCE))

    def test_09_npy_tamper_rejected(self):
        item = next(x for x in self.loader.contents.values() if x['kind'] == 'rgb')
        blob = bytearray(l.safe_path(LINE, item['line_relative_path']).read_bytes())
        blob[-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'PIXEL_HASH_OR_LENGTH'):
            l.npy_pixels(bytes(blob), item['raw_pixel_sha256'], 'rgb')

    def test_10_future_policy_noninterference(self):
        trace = copy.deepcopy(next(iter(self.grids[0].values())))
        task = next(iter(self.compiler.tasks))
        before = self.compiler.policy_at(trace, task, 206, 'same')
        trace['observations'][-1]['rgb_hash'] = 'f'*64
        trace['observations'][-1]['pixels'] = {}
        self.assertEqual(before, self.compiler.policy_at(trace, task, 206, 'same'))

    def test_11_query_permutation_rejected(self):
        query = copy.deepcopy(self.loader.cells[0]['query'])
        query['sequence'][0], query['sequence'][1] = query['sequence'][1], query['sequence'][0]
        with self.assertRaisesRegex(ValueError, 'QUERY_ORDER'):
            self.compiler.encode_query(query)

    def test_12_source_seal_all_files(self):
        count = 0
        for line in (SOURCE / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            self.assertEqual(e.sha(l.safe_path(SOURCE, relative).read_bytes()), expected)
            count += 1
        self.assertEqual(count, 987)

    def test_13_compiler_config_mismatch_rejected(self):
        tasks = {k: dict(v) for k, v in self.compiler.tasks.items()}
        tasks[next(iter(tasks))]['instruction'] += ' changed'
        wrong = c.Compiler(self.compiler.roles, tasks, self.compiler.eligible)
        with self.assertRaisesRegex(ValueError, 'COMPILER_CONFIG_MISMATCH'):
            l.FamilyLoader(self.root, wrong)

    def test_14_missing_house_rejected_before_write(self):
        candidate = copy.deepcopy(self.candidate)
        candidate.pop('context')
        with self.assertRaisesRegex(ValueError, 'HOUSE_GROUP_REQUIRED'):
            e.export_family(Path(self.temp.name) / 'nohouse', self.compiler, candidate, self.grids[0],
                            SOURCE / 'content', family_id='nohouse')

    def test_15_independent_full_supervision_readback(self):
        result = self.loader.validate_supervision_contract()
        self.assertEqual(result['labels_recomputed'], 18)
        self.assertEqual(result['ce_owners_verified'], 1410)


if __name__ == '__main__':
    unittest.main(verbosity=2)
