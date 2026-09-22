import gzip,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
class Tests(unittest.TestCase):
    def fixture(self,root):
        e=dict(episode_id=1,instruction='go',steps=2,stopped=True,termination='STOP',positions=[[0,0,0],[1,0,0],[1,0,0]],distances=[3.,2.,2.],success=1.,spl=1.,ndtw=1.,oracle_success=1.,mode='POLICY')
        u.write(root/'episode_00.json',e)
        u.append(root/'INTERFACE.jsonl',dict(rgb_sha256='rgb0'))
        for i,(act,rgb,history,window,z) in enumerate([('move_forward','rgb1',[],['rgb0'],[3,0,0,-1]),('STOP','rgb1',['move_forward'],['rgb0','rgb1'],[0,0,0,3])],1):
            u.append(root/'POLICY_STEPS.jsonl',dict(step=i,executed_action=act,logits=z,raw=dict(instruction='go',executed_history=history,rgb_sha256=window)))
            u.append(root/'STEPS_PRIVILEGED.jsonl',dict(step=i,action=act,position=[1,0,0],distance_to_goal=2.,rgb_sha256=rgb))
        gt=root/'truth.json.gz'
        with gzip.open(gt,'wt') as f:json.dump({'1':dict(locations=[[0,0,0],[1,0,0]])},f)
        return e,gt
    def test_actual_actions_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,gt=self.fixture(root);self.assertEqual(u.audit_episode(root,0,str(gt))['success'],1)
    def test_gt_split_is_not_reused_by_episode_id(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,gt=self.fixture(root);u.audit_episode(root,0,str(gt));other=root/'unseen.json.gz'
            with gzip.open(other,'wt') as f:json.dump({'1':dict(locations=[[9,0,0],[10,0,0]])},f)
            with self.assertRaisesRegex(AssertionError,'METRIC_RECOMPUTATION'):u.audit_episode(root,0,str(other))
    def test_teacher_interruption_not_admitted_as_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,gt=self.fixture(root);e.update(stopped=False,termination='TEACHER_UNAVAILABLE',success=0.,spl=0.);u.write(root/'episode_00.json',e)
            with self.assertRaisesRegex(AssertionError,'NONSTOP_EARLY_END'):u.audit_episode(root,0,str(gt))
    def test_history_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);e,gt=self.fixture(root);p=root/'POLICY_STEPS.jsonl';rows=u.c.records(p);rows[1]['raw']['executed_history']=['turn_left'];p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            with self.assertRaisesRegex(AssertionError,'HISTORY'):u.audit_episode(root,0,str(gt))
    def test_frozen_window_rejects_privileged_fields(self):
        with self.assertRaises(ValueError):u.c.Window().receive(dict(done=True,distance=2))
    def test_budget_stop_counts(self):self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
    def test_real_split_and_registry(self):
        m=u.HERE/'manifests';s=u.read(m/'SPLIT.json')
        self.assertEqual(len(s['all_training_houses']),40);self.assertFalse(set(s['all_training_houses'])&set(s['dev_houses']+s['unseen_houses']))
        self.assertEqual(len(u.read(m/'FIT_ORDER.json')),640);self.assertEqual(len(u.read(m/'DEV_ORDER.json')),100);self.assertEqual(len(u.read(m/'UNSEEN_ORDER.json')),1839)
        self.assertFalse(u.read(m/'BASE_TRAIN_SCENE_AUDIT.json')['val_unseen_overlap'])
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures),scope='CPU transport, split, metric and budget contracts; no model-quality claim'));raise SystemExit(0 if r.wasSuccessful() else 1)
