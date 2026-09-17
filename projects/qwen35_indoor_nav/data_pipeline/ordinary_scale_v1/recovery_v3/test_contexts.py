import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v3_supervisor',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

class ContextTests(unittest.TestCase):
    def test_new_small_context_allowed(self):
        g=dict(memory_mib=1900,processes={1:246,2:246,3:500,4:768})
        self.assertEqual(len(run.validate_contexts(g,None)),4)
    def test_large_external_rejected(self):
        with self.assertRaises(AssertionError):
            run.validate_contexts(dict(memory_mib=900,processes={1:769}),None)
    def test_total_external_rejected(self):
        with self.assertRaises(AssertionError):
            run.validate_contexts(dict(memory_mib=2400,processes={1:768,2:768,3:768}),None)
    def test_own_count_separated(self):
        self.assertEqual(run.validate_contexts(dict(memory_mib=4300,processes={1:3800,2:500}),1),{2:500})
    def test_own_budget_rejected(self):
        with self.assertRaises(AssertionError):
            run.validate_contexts(dict(memory_mib=5000,processes={1:4096,2:500}),1)

if __name__=='__main__': unittest.main()
