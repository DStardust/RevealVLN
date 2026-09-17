import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('common');metrics=load('metrics')

class Sim:
    def __init__(self): self.p=np.array([0.,0.,0.])
    def get_agent_state(self): return NS(position=self.p.copy())
    def geodesic_distance(self,p,goals,episode): return float(np.linalg.norm(p-goals[0]))

class Tests(unittest.TestCase):
    def test_action_map(self):
        self.assertEqual(dict(zip(c.ACTIONS,c.HABITAT_IDS)),dict(move_forward=1,turn_left=2,turn_right=3,STOP=0))
    def test_stop_consumes_step(self):
        self.assertEqual(c.advance('STOP',499,False),(500,True))
    def test_motion_limit(self):
        self.assertEqual(c.advance('move_forward',499,False),(500,True))
        with self.assertRaises(ValueError):c.advance('STOP',500,True)
    def test_after_stop(self):
        with self.assertRaises(ValueError):c.advance('move_forward',1,True)
    def test_quaternion(self):
        self.assertEqual(c.xyzw_to_wxyz([1,2,3,4]),(4,1,2,3))
    def test_short_window_reset(self):
        blob=base64.b64encode(bytes(224*224*3)).decode();w=c.Window()
        w.receive(dict(done=False,rgb=blob,instruction='first'))
        for _ in range(10):w.receive(dict(done=False,rgb=blob),executed='move_forward')
        self.assertEqual((len(w.images),len(w.executed)),(2,8))
        w.receive(dict(done=False,rgb=blob,instruction='second'))
        self.assertEqual((len(w.images),len(w.executed),w.instruction),(1,0,'second'))
    def test_future_fields_rejected(self):
        blob=base64.b64encode(bytes(224*224*3)).decode()
        for key in ('scene_id','goal','distance','future_rgb','oracle_action'):
            with self.assertRaises(ValueError):
                c.Window().receive(dict(done=False,rgb=blob,instruction='go',**{key:123}))
    def test_terminal_has_no_truth(self):
        self.assertFalse(c.Window().receive(dict(done=True)))
        with self.assertRaises(ValueError):c.Window().receive(dict(done=True,success=1))
    def test_bad_rgb(self):
        with self.assertRaises(ValueError):c.Window().receive(dict(done=False,rgb='eA==',instruction='go'))
    def test_success_requires_stop(self):
        s=Sim();m=metrics.OfficialMetrics(s,[10.,0.,0.]);s.p=np.array([8.,0.,0.])
        self.assertEqual(m.update(False)['success'],0)
        self.assertEqual(m.update(True)['success'],1)
        self.assertEqual(m.get()['spl'],1)
    def test_strict_three_meters(self):
        s=Sim();m=metrics.OfficialMetrics(s,[10.,0.,0.]);s.p=np.array([7.,0.,0.])
        self.assertEqual(m.update(True)['success'],0)
    def test_distant_stop(self):
        s=Sim();m=metrics.OfficialMetrics(s,[10.,0.,0.])
        self.assertEqual(m.update(True)['success'],0)
        self.assertEqual(m.get()['spl'],0)
    def test_spl_detour(self):
        s=Sim();m=metrics.OfficialMetrics(s,[10.,0.,0.]);s.p=np.array([0.,0.,10.]);m.update(False)
        s.p=np.array([8.,0.,0.]);v=m.update(True)
        self.assertAlmostEqual(v['spl'],10/(10+np.sqrt(164)),places=12)
    def test_repeated_position(self):
        s=Sim();m=metrics.OfficialMetrics(s,[10.,0.,0.]);s.p=np.array([8.,0.,0.]);m.update(False)
        for _ in range(5):m.update(False)
        self.assertEqual(m.update(True)['spl'],1)

if __name__=='__main__':unittest.main()
