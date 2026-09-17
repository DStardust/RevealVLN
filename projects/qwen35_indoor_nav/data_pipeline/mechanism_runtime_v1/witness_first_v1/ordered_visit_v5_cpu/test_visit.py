import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import copy
import unittest
from visit import evaluate

ELIGIBLE={'A':['a'],'B':['b']}
def frame(location):
    return dict(evidence_verified=True,**{r:dict(zone_geodesic_m=0.5 if r==location else 4.0,
        visible_instance_pixels={r.lower():400} if r==location else {},zone_certified=True) for r in ELIGIBLE})
def trace(locations):
    return dict(observations=[frame(x) for x in locations],actions=['F']*(len(locations)-2)+['S'],
                complete=True,collisions=0,zones_disjoint_verified=True)
class Tests(unittest.TestCase):
    def test_order(self):
        t=trace(['A','A','B','B','B']);self.assertEqual(evaluate(t,ELIGIBLE)['y'],1)
    def test_reverse_and_missed(self):
        for locations in (['B','B','A','A'],['B','B','B','B'],['A','A','X','X']):
            self.assertEqual(evaluate(trace(locations),ELIGIBLE)['y'],0)
    def test_return_to_b_after_a(self):
        self.assertEqual(evaluate(trace(['B','B','A','A','B','B','B']),ELIGIBLE)['y'],1)
    def test_stop_away(self):
        self.assertEqual(evaluate(trace(['A','A','B','B','X','X']),ELIGIBLE)['y'],0)
    def test_visible_but_far(self):
        t=trace(['A','A','B','B','B'])
        for o in t['observations']:o['A']['zone_geodesic_m']=4.0
        self.assertEqual(evaluate(t,ELIGIBLE)['y'],0)
    def test_unknown_never_negative(self):
        base=trace(['A','A','B','B','B'])
        for key,value in [('complete',False),('collisions',1),('zones_disjoint_verified',False)]:
            t=copy.deepcopy(base);t[key]=value;self.assertIsNone(evaluate(t,ELIGIBLE)['y'])
        t=copy.deepcopy(base);t['observations'][0]['A']['zone_geodesic_m']=float('nan')
        self.assertIsNone(evaluate(t,ELIGIBLE)['y'])
    def test_same_instance_two_frames(self):
        t=trace(['A','A','B','B','B']);t['observations'][0]['A']['visible_instance_pixels']={'other':400}
        self.assertEqual(evaluate(t,ELIGIBLE)['y'],0)
    def test_overlap_unknown(self):
        t=trace(['A','A','B','B','B'])
        for o in t['observations'][:2]:o['B']=frame('B')['B']
        self.assertIsNone(evaluate(t,ELIGIBLE)['y'])
    def test_not_immediate_stop_or_input_truth(self):
        t=trace(['A','A','B','B','B','B']);before=copy.deepcopy(t);r=evaluate(t,ELIGIBLE)
        self.assertEqual(r['y'],1);self.assertFalse(r['policy_input_allowed']);self.assertEqual(t,before)
if __name__=='__main__':unittest.main()
