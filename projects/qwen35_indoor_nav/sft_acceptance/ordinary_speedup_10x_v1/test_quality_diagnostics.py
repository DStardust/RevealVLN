import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('quality',Path(__file__).parent/'quality_diagnostics.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
def row(n,ce,m):return dict(cursor=dict(decisions=n),metrics=dict(mean_ce=ce,confusion=m))
class Tests(unittest.TestCase):
    def test_exact_window_and_constant_baseline(self):
        a=row(4,2.,[[2,0,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,0]])
        b=row(8,1.5,[[4,0,0,0],[1,0,0,0],[1,0,1,0],[1,0,0,0]])
        r=q.summarize([a,b],4);self.assertEqual(r['mean_ce'],1.);self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['always_forward_accuracy'],.5);self.assertEqual(r['recall'],[1.,None,0.,0.])
        self.assertEqual(r['prediction_counts'],[4,0,0,0])
    def test_no_records_not_zero(self):self.assertFalse(q.summarize([])['available'])
    def test_bad_counters(self):self.assertFalse(q.valid(row(9,1.,[[0]*4 for _ in range(4)])))
    def test_reset_not_negative_window(self):
        a=row(4,1.,[[4,0,0,0],[0]*4,[0]*4,[0]*4]);b=row(2,1.,[[2,0,0,0],[0]*4,[0]*4,[0]*4])
        self.assertEqual(q.summarize([a,b])['reason'],'COUNTER_RESET')
    def test_nonfinite_or_invalid_shapes(self):
        for r in [None,{},dict(cursor=[],metrics={}),row(1,float('nan'),[[1,0,0,0],[0]*4,[0]*4,[0]*4])]:self.assertFalse(q.valid(r))
if __name__=='__main__':unittest.main()
