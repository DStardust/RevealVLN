from pathlib import Path
import sys
import types
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import recover


class ResourceTests(unittest.TestCase):
    def fixture(self):
        return dict(error="AssertionError('SHARD_WALL_BUDGET')",workers=[dict(shard=0,returncode=1)]),{0:dict(gpu=3,shard=0,pid=123)}
    def test_known_budget_closed_worker_only(self):
        result,processes=self.fixture()
        recover.validate_resource_workers(result,processes,[0],3,lambda pid:False)
        result['error']="AssertionError('LANE_WALL_BUDGET')"
        recover.validate_resource_workers(result,processes,[0],3,lambda pid:False)
    def test_unknown_error_rejected(self):
        result,processes=self.fixture()
        for error in [None,"AssertionError('EXTERNAL_GPU_RESOURCE')","RuntimeError('unknown')","AssertionError('SHARD_WALL_BUDGET'); OWN_CLEANUP: error"]:
            result['error']=error
            with self.assertRaises(AssertionError):recover.validate_resource_workers(result,processes,[0],3,lambda pid:False)
    def test_present_pid_rejected(self):
        result,processes=self.fixture()
        with self.assertRaises(AssertionError):recover.validate_resource_workers(result,processes,[0],3,lambda pid:True)
    def test_missing_or_mismatched_worker_rejected(self):
        result,processes=self.fixture()
        with self.assertRaises(AssertionError):recover.validate_resource_workers(result,processes,[1],3,lambda pid:False)
        processes[0]['gpu']=4
        with self.assertRaises(AssertionError):recover.validate_resource_workers(result,processes,[0],3,lambda pid:False)
    def test_unknown_exit_rejected(self):
        result,processes=self.fixture()
        for code in [None,True,137,2]:
            result['workers'][0]['returncode']=code
            with self.assertRaises(AssertionError):recover.validate_resource_workers(result,processes,[0],3,lambda pid:False)
    def test_scope_and_no_quality_override(self):
        self.assertEqual(set(recover.EXPECTED_RUNTIME_LOCKS),{3,4})
        source=recover.recovery_source();compile(source,'CPU_resource_recovery','exec')
        self.assertNotIn("c.state(shard)['complete']",source)
        self.assertIn("cmdline']==['']",source)
        self.assertIn('full_dataset_certified=False',source)
        self.assertIn('ops.contexts(snapshot,restorable=True)',source)


def suite():
    text=recover.source_file('test_recover.py')
    text=recover.exact(text,'HERE=Path(__file__).resolve().parent',f'HERE=Path({str(HERE)!r})')
    module=types.ModuleType('inherited_identity_safety');module.__file__=str(HERE/'test_resource_recovery.py')
    exec(compile(text,module.__file__,'exec'),module.__dict__)
    names=[n for n in unittest.defaultTestLoader.getTestCaseNames(module.Tests) if n!='test_scope_known_four_and_naturalclose']
    return unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ResourceTests)]+
        [module.Tests(name) for name in names])


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(suite());raise SystemExit(not r.wasSuccessful())
