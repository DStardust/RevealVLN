"""Stdlib scheduling, bounds and normalization regression tests."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('prep',HERE/'prepare.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)

class Tests(unittest.TestCase):
    def setUp(self):self.rows=[dict(decisions=i+5) for i in range(10)]
    def test_determinism(self):self.assertEqual(p.schedule(self.rows,0),p.schedule(self.rows,0))
    def test_rank_isolation(self):
        a=p.schedule(self.rows,0);b=p.schedule(self.rows,1)
        self.assertFalse({x['row'] for x in a}&{x['row'] for x in b})
    def test_bound(self):
        for rank in range(2):
            ss=p.schedule(self.rows,rank)
            self.assertEqual(len(ss),800)
            self.assertTrue(all(0<x['end']-x['start']<=4 for x in ss))
    def test_causal_contiguity(self):
        for rank in range(2):
            ss=p.schedule(self.rows,rank)
            for a,b in zip(ss,ss[1:]):
                if b['start']:
                    self.assertEqual(a['row'],b['row']);self.assertEqual(a['end'],b['start'])
                else:self.assertEqual(a['end'],self.rows[a['row']]['decisions'])
    def test_global_normalization(self):
        a,b,n,m=13.,19.,3,7
        ddp_mean=(a+b)/2
        self.assertAlmostEqual(ddp_mean*2/(n+m),(a+b)/(n+m))
    def test_budget(self):
        ss=[p.schedule(self.rows,r) for r in range(2)]
        for update in range(200):
            self.assertLessEqual(sum(x['end']-x['start'] for rank in ss for x in rank[update*4:update*4+4]),32)

if __name__=='__main__':unittest.main()
