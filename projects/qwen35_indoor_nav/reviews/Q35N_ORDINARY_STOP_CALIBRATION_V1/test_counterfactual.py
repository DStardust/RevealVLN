import importlib.util
import math
from pathlib import Path
import unittest

s=importlib.util.spec_from_file_location('prefix_test',Path(__file__).with_name('counterfactual.py'))
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


class PrefixTests(unittest.TestCase):
    def episode(self):
        e=dict(index=0,episode_id=42,house='h',steps=3,stopped=True,
               positions=[[0,0,0],[1,0,0],[2,0,0],[2,0,0]],distances=[4.,3.,2.,2.])
        ds=[dict(index=0,step=1,action='move_forward',logits=[2.,0.,0.,0.]),
            dict(index=0,step=2,action='move_forward',logits=[1.,0.,0.,.9]),
            dict(index=0,step=3,action='STOP',logits=[0.,0.,0.,1.])]
        return e,ds,[[0,0,0],[1,0,0],[2,0,0]]
    def test_zero_identity(self):
        e,d,r=self.episode();p=c.prefix(e,d,r,0)
        self.assertEqual((p['steps'],p['success'],p['spl'],p['ndtw']),(3,1.,1.,1.))
    def test_stop_uses_preaction_position_and_strict_radius(self):
        e,d,r=self.episode();p=c.prefix(e,d,r,.25)
        self.assertEqual((p['steps'],p['navigation_error_m'],p['success']),(2,3.,0.))
        self.assertEqual(p['path_length_m'],1.)
    def test_first_stop_only(self):
        e,d,r=self.episode();p=c.prefix(e,d,r,3.)
        self.assertEqual((p['steps'],p['path_length_m']),(1,0.))
    def test_tie_precedence(self):
        self.assertEqual(c.choose([1,1,1,0],1),'move_forward')
        self.assertEqual(c.choose([0,1,1,0],1),'turn_left')
    def test_validation(self):
        for logits,b in [([0,0,0,0],-1),([0,0,0,0],math.nan),([0,0,0,math.inf],0),([0],0)]:
            with self.assertRaises(ValueError):c.choose(logits,b)
    def test_raw_action_tampering_rejected(self):
        e,d,r=self.episode();d[0]['action']='STOP'
        with self.assertRaises(AssertionError):c.prefix(e,d,r,0)
    def test_no_stop_keeps_budget(self):
        e=dict(index=0,episode_id=42,house='h',steps=500,stopped=False,
               positions=[[0,0,0]]*501,distances=[5.]*501)
        ds=[dict(index=0,step=j+1,action='turn_left',logits=[0,3,0,0]) for j in range(500)]
        p=c.prefix(e,ds,[[0,0,0]],2.)
        self.assertEqual((p['steps'],p['success'],p['stopped']),(500,0.,False))
    def test_decision_signature_has_no_oracle(self):
        import inspect
        self.assertEqual(list(inspect.signature(c.choose).parameters),['logits','bias'])
    def test_grid_not_expandable(self):
        with self.assertRaises(AssertionError):c.select([])
    def test_nonstop_actions_never_changed(self):
        import random
        rng=random.Random(1309)
        for _ in range(1000):
            logits=[rng.uniform(-5,5) for _ in range(4)]
            for b in c.GRID:
                a=c.choose(logits,b)
                self.assertTrue(a=='STOP' or a==c.choose(logits,0))
    def test_duplicate_positions_do_not_change_ndtw(self):
        self.assertEqual(c.ndtw([[0,0,0],[0,0,0],[1,0,0]],[[0,0,0],[1,0,0]]),1.)
    def test_selection_tie_prefers_small_bias(self):
        rows=[dict(bias=b,comparison=dict(fit_gate=b>0),metrics=dict(sr=.25,spl=.2)) for b in c.GRID]
        self.assertEqual(c.select(rows)['bias'],.25)
        for row in rows:row['comparison']['fit_gate']=False
        self.assertIsNone(c.select(rows))
    def test_two_house_net_gate(self):
        a=[dict(index=i,episode_id=i,house='h'+str(i),success=0.,spl=0.,ndtw=.5) for i in range(4)]
        b=[dict(x,success=float(i<2),spl=float(i<2)) for i,x in enumerate(a)]
        self.assertTrue(c.assess(a,b)['fit_gate'])
        b[1].update(success=0.,spl=0.)
        self.assertFalse(c.assess(a,b)['fit_gate'])


if __name__=='__main__':unittest.main()
