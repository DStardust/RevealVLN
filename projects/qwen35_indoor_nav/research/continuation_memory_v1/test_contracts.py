"""CPU boundary tests against a real existing export; no model, autograd or replay."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s = load('continuation_schema', HERE/'schema.py')
check = load('continuation_exact_state', HERE/'check_tasks.py')
core = load('continuation_frozen_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
loaders = load('continuation_frozen_loader', LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
EXPORT = LINE/'data_pipeline/mechanism_runtime_v1/witness_first_v1/short_revisit_v3/run_v1/bundles/WF_SHORT_REVISIT_V3_004/export_v4'


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = json.loads((EXPORT/'MANIFEST.json').read_text())
        cls.compiler = core.Compiler(**manifest['compiler_config'])
        cls.loader = loaders.FamilyLoader(EXPORT, cls.compiler)
        cls.records = cls.loader.prefix_records(next(iter(cls.loader.prefix_index)))

    def test_real_loader_to_policy_has_only_causal_fields(self):
        obs = s.from_existing_v4(self.loader, self.records[-1])
        self.assertEqual(set(s.policy_arguments(obs, None, memory_instruction=obs.instruction)),
                         {'instruction','rgb','executed','memory'})
        self.assertEqual([len(x) for x in obs.rgb], [150528, 150528])

    def test_future_query_ids_and_pose_rejected_by_policy(self):
        obs = s.from_existing_v4(self.loader, self.records[-1])
        for field in ('query','y','pose','family_id','task_state','future_rgb'):
            with self.assertRaises(TypeError):
                s.PolicyObservation(obs.instruction, obs.rgb, obs.executed, **{field: 'forbidden'})

    def test_transport_pose_injection_is_rejected(self):
        record = copy.deepcopy(self.records[-1]); record['pose'] = [0,0,0]
        with self.assertRaises(ValueError):s.from_existing_v4(self.loader, record)

    def test_task_change_cannot_reuse_task_conditioned_memory(self):
        obs = s.from_existing_v4(self.loader, self.records[-1])
        with self.assertRaisesRegex(ValueError,'TASK_CHANGE'):
            s.policy_arguments(obs, None, memory_instruction='another instruction')

    def test_real_prefix_sequence_and_reset(self):
        observations = tuple(s.from_existing_v4(self.loader, r) for r in self.records[:4])
        self.assertEqual(len(s.PrefixTrace(observations).observations),4)
        with self.assertRaises(ValueError):s.PrefixTrace(observations[1:])
        with self.assertRaises(ValueError):s.PrefixTrace((observations[0],observations[2]))

    def test_stop_cannot_be_written_as_past_motion(self):
        obs = s.from_existing_v4(self.loader, self.records[-1])
        with self.assertRaises(ValueError):s.PolicyObservation(obs.instruction, obs.rgb, ('STOP',))

    def test_unknown_mask_is_preserved(self):
        s.LabelCertificate('UNKNOWN',None,0)
        for y,mask in ((0,1),(0,0),(1,1)):
            with self.assertRaises(ValueError):s.LabelCertificate('UNKNOWN',y,mask)

    def test_real_queries_require_semantic_projection_and_bound_vocabulary(self):
        query = self.loader.cells[0]['query']
        with self.assertRaises(ValueError):s.continuation_query(self.compiler,query)
        clean = self.compiler.semantic_query(query)
        result = s.continuation_query(self.compiler,clean)
        self.assertIn('vocabulary',result)
        clean['sequence'][0]['family_id']='forbidden'
        with self.assertRaises(ValueError):s.continuation_query(self.compiler,clean)

    def test_exact_state_matches_all_existing_task_traces(self):
        for cell in self.loader.cells:
            trace=self.loader.read(cell['trace_path'])
            rows=check.exact_state_targets(self.compiler,trace,cell['task_id'],cell['prefix_cutoff'])
            old=self.compiler.m2(trace,cell['task_id'])[:cell['prefix_cutoff']+1]
            self.assertEqual([r['ordered_ready_to_stop'] for r in rows],
                             [r['state']=='READY_TO_STOP' if r['truth_known'] else None for r in old])

    def test_future_corruption_cannot_change_prefix_state(self):
        cell=self.loader.cells[0]; trace=self.loader.read(cell['trace_path']);cut=cell['prefix_cutoff']
        expected=check.exact_state_targets(self.compiler,trace,cell['task_id'],cut)
        trace['observations'][cut+1:]=[{'invalid':'future'}]
        trace['actions'][cut:]=['INVALID']
        self.assertEqual(check.exact_state_targets(self.compiler,trace,cell['task_id'],cut),expected)

    def test_missing_evidence_is_unknown_not_negative(self):
        cell=self.loader.cells[0];trace=self.loader.read(cell['trace_path'])
        trace['observations'][0]['evidence_complete']=False
        self.assertEqual(self.compiler.evaluate(trace,cell['task_id']),'unknown')
        state=check.exact_state_targets(self.compiler,trace,cell['task_id'],2)[-1]
        self.assertEqual(state['loss_mask'],0); self.assertIsNone(state['ordered_ready_to_stop'])

    def test_house_and_history_continuation_cross_split_grouping(self):
        a=s.FamilySplit('train','house_a','family_a',('route_a',),('h_a',),('c_a',),('language_a',),'already_exposed')
        b=s.FamilySplit('dev','house_b','family_b',('route_b',),('c_a',),('c_b',),('language_b',),'already_exposed')
        with self.assertRaises(ValueError):s.validate_splits([a,b])
        b=s.FamilySplit('dev','house_a','family_b',('route_b',),('h_b',),('c_b',),('language_b',),'already_exposed')
        with self.assertRaises(ValueError):s.validate_splits([a,b])

    def test_exposed_data_cannot_be_renamed_to_independent_test(self):
        with self.assertRaises(ValueError):
            s.FamilySplit('test','house','family',('r',),('h',),('c',),('l',),'already_exposed')

    def test_auxiliary_weights_equalize_families(self):
        yes=s.LabelCertificate('PASS',1,1); no=s.LabelCertificate('FAIL',0,1); unknown=s.LabelCertificate('UNKNOWN',None,0)
        weights=s.family_aux_weights([[yes,no],[yes]*10+[unknown],[unknown]])
        self.assertAlmostEqual(sum(weights[0]),.5);self.assertAlmostEqual(sum(weights[1]),.5)
        self.assertEqual(weights[1][-1],0.);self.assertEqual(weights[2],[0.])
        with self.assertRaises(ValueError):s.family_aux_weights([[unknown]])


if __name__=='__main__':
    with (HERE/'CPU_TEST_LOG.txt').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        scope='Existing real export and CPU information/state contracts only',
        model_implemented=False,cross_step_gradient_tested=False,new_gpu_starts=0,new_physical_replays=0,
        research_status='RESEARCH_HYPOTHESIS_UNTESTED')
    with (HERE/'CPU_TEST_RESULT.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
