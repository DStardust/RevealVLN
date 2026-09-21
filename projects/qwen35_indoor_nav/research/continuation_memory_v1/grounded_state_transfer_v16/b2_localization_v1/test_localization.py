import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from common import fit_weights,metrics,age_bucket,merge_teacher
from evaluator_v16 import legacy,state_sequence

class Contract(unittest.TestCase):
    def test_two_legal_teacher_alternatives_retained(self):
        old=dict(state=[0]*4,teacher_targets=[1],action_mask=1,preservation=True)
        merge_teacher(old,dict(state=[0]*4,teacher_targets=[2],action_mask=1,preservation=False))
        self.assertEqual(old['teacher_targets'],[1,2])
        with self.assertRaises(ValueError):merge_teacher(old,dict(state=[1]*4))
    def test_no_DEV_optimization(self):
        with self.assertRaises(ValueError):fit_weights([dict(split='DEV',parent='x')],torch.tensor([[1.]]))
    def test_parent_duplicate_does_not_increase_weight(self):
        rows=[dict(split='FIT',parent=p) for p in ('a','a','b','b')];y=torch.tensor([[0.],[1.],[0.],[1.]])
        a=fit_weights(rows,y);b=fit_weights(rows+rows[:2],torch.cat([y,y[:2]]))
        self.assertAlmostEqual(float(a[:2].sum()/a.sum()),float(torch.cat([b[:2],b[4:]]).sum()/b.sum()))
    def test_present_event_is_distinct_from_past_state(self):
        compiler=legacy.Compiler({'anchor':['plant','room'],'terminal':['sink','room']},{'A':dict(anchor='anchor',terminal='terminal',instruction='plant then sink')},{'anchor':[1],'terminal':[2]})
        obs=[dict(step=t,evidence_complete=True,pixels={'1':256 if t<2 else 0,'2':0}) for t in range(4)]
        self.assertTrue(compiler.atoms(obs)[1]['anchor']);self.assertFalse(compiler.atoms(obs)[3]['anchor'])
        self.assertEqual(state_sequence(compiler,obs,'A')[3][1],1)
    def test_no_single_class_balanced_score(self):
        self.assertIsNone(metrics([1,1],[.9,.8])['balanced_accuracy'])
        self.assertEqual(metrics([1,0],[.9,.8])['balanced_accuracy'],.5)
    def test_age_no_event_not_zero(self):
        self.assertEqual(age_bucket(None),'never');self.assertEqual(age_bucket(0),'current');self.assertEqual(age_bucket(19),'17+')
if __name__=='__main__':unittest.main()
