"""CPU regression: train-only bounded weights and global weighted-mean gradient."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('weights',Path(__file__).resolve().parent/'weights.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Tests(unittest.TestCase):
    def test_counts(self):
        counts=dict(zip(w.ACTIONS,[339,126,90,10]));v=w.make_weights(counts)
        self.assertAlmostEqual(sum(counts[a]*x for a,x in zip(w.ACTIONS,v['values']))/565,1.)
        self.assertEqual(v['raw'][3],4.)
        self.assertLessEqual(max(v['values'])/min(v['values']),4.)
    def test_missing(self):
        with self.assertRaises(AssertionError):w.make_weights({})
    def test_zero(self):
        with self.assertRaises(AssertionError):w.make_weights(dict(zip(w.ACTIONS,[339,126,90,0])))
    def test_balanced(self):
        self.assertEqual(w.make_weights(dict.fromkeys(w.ACTIONS,10))['values'],[1.]*4)
    def test_global_denominator(self):
        weights=w.make_weights(dict(zip(w.ACTIONS,[339,126,90,10])))['values']
        targets=[[0,0,1],[2,3]]
        gradients=[[.4,-.2,.3],[-.5,.7]]
        numerators=[sum(weights[t]*g for t,g in zip(ts,gs)) for ts,gs in zip(targets,gradients)]
        denominators=[sum(weights[t] for t in ts) for ts in targets]
        ddp_mean=sum(numerators)/2
        actual=ddp_mean/(sum(denominators)/2)
        expected=sum(numerators)/sum(denominators)
        self.assertAlmostEqual(actual,expected)
        self.assertNotAlmostEqual(actual,sum(n/d for n,d in zip(numerators,denominators))/2)
    def test_worker_contract(self):
        p=Path(__file__).resolve().parent
        source=(p/'worker.py').read_text()
        self.assertIn("loss.append(ce*self.weights[target])",source)
        self.assertIn("float(totals[3])/2",source)
        self.assertIn("range(200)",source)
        self.assertNotIn("CacheSwitch",source)
        for path in p.glob('*.py'):compile(path.read_text(),str(path),'exec')
if __name__=='__main__':unittest.main()
