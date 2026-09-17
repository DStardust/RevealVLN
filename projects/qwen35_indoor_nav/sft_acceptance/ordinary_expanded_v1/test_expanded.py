import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('test_expanded_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('reuse'); d=load('data'); c=load('control')


class Expanded(unittest.TestCase):
    def test_frozen_parents(self):
        for name,digest in r.HASHES.items(): self.assertEqual(hashlib.sha256((r.OLD/name).read_bytes()).hexdigest(),digest)

    def test_compiled_sources(self):
        for name in r.HASHES: compile(r.source(name),name,'exec')

    def test_model_exact(self): self.assertEqual(r.source('model.py'),(r.OLD/'model.py').read_text())
    def test_control_exact(self): self.assertEqual(r.source('control.py'),(r.OLD/'control.py').read_text())
    def test_launcher_exact(self): self.assertEqual(r.source('launcher.py'),(r.OLD/'launcher.py').read_text())

    def test_train_only_registered_transform(self):
        a=r.source('train_filestore.py').replace('Q35N_ORDINARY_EXPANDED_V1','Q35N_ORDINARY_SYNC_RECOVERY_V1')
        a=a.replace("HERE / 'SAMPLE_INDEX.jsonl'", "HERE.parent / 'ordinary_baseline_v3/SAMPLE_INDEX.jsonl'")
        a=a.replace('legacy_charge_includes_conservative_reserve=False','legacy_charge_includes_conservative_reserve=True')
        self.assertEqual(a,(r.OLD/'train_filestore.py').read_text())

    def test_pure_batch_coverage(self):
        samples=[dict(est=170+i%100) for i in range(503)]
        p=d.plan_epoch_batches(samples,6144,1209,0,3)
        self.assertEqual(len({len(x) for x in p}),1)
        flat=[i for rank in p for batch in rank for i in batch]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(p,d.plan_epoch_batches(samples,6144,1209,0,3))

    def test_inflection_and_stop(self):
        acts=['move_forward','move_forward','turn_left','STOP']
        self.assertEqual([d.inflection_weight(acts,i,3.2) for i in range(4)],[3.2,1.,3.2,3.2])

    def test_terminal_epoch(self):
        self.assertEqual(d.advance_epoch_boundary(dict(epoch=0,position=10,updates=10,decisions=24),10)['epoch'],1)

    def test_policy_whitelist(self):
        class Record:
            def decision(self,t): return dict(control=dict(decision_step=t),policy=dict(instruction='go',images=[],executed_actions=[]),supervision=dict(target_action='STOP'))
        store=d.SampleStore.__new__(d.SampleStore);store._cache={0:Record()}
        self.assertEqual(set(store.get(0,0)),{'instruction','images','executed','target'})

    def test_metadata_rejected_from_policy(self):
        class Bad:
            def decision(self,t): return dict(control=dict(decision_step=t),policy=dict(instruction='go',images=[],executed_actions=[],goal_position=[0,0,0]),supervision=dict(target_action='STOP'))
        store=d.SampleStore.__new__(d.SampleStore);store._cache={0:Bad()}
        with self.assertRaisesRegex(ValueError,'MODEL_WHITELIST'):store.get(0,0)

    def test_no_auto_retry(self):
        code=r.source('supervise_filestore.py')
        self.assertIn('range(1, 2)',code);self.assertIn('max_attempts=1',code)


if __name__=='__main__': unittest.main()
