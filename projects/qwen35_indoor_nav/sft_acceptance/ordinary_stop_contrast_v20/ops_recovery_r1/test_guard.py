import importlib.util,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('guard',HERE/'guard.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Test(unittest.TestCase):
    def test_exact_infrastructure_failure_only(self):
        a=dict(status='STOPPED_WITH_EVIDENCE',reason="TimeoutError('DISK_SCAN_DEADLINE')")
        self.assertTrue(g.eligible(a,'failed',False));self.assertFalse(g.eligible(a,'active',False));self.assertFalse(g.eligible(a,'failed',True));self.assertFalse(g.eligible(dict(a,reason='BAD_SCORE'),'failed',False));self.assertFalse(g.eligible(dict(a,reason="ValueError('PREFIX_INPUT_MISMATCH')"),'failed',False))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Test));(HERE/'GUARD_TEST_RESULT.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),scope='Only exact resource failure eligible; active/completed/numeric/score failures never trigger retry'),indent=2)+'\n');raise SystemExit(not r.wasSuccessful())
