import importlib.util,json,unittest
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('rgb',Path(__file__).with_name('rgb.py'));rgb=importlib.util.module_from_spec(s);s.loader.exec_module(rgb)
class Tests(unittest.TestCase):
    def test_constant_observation_retained(self):
        x=np.zeros((224,224,3),dtype=np.uint8);r=rgb.validate(x);self.assertTrue(r['uniform']);self.assertEqual(r['sha256'],'0a3f0ee9e3cbab26f89ed53b7b20e22cb985b650fd4c52ef57e4aeb27560d773')
    def test_nonuniform_bytes_unchanged(self):
        x=np.zeros((224,224,3),dtype=np.uint8);x[0,0]=42;before=x.copy();self.assertFalse(rgb.validate(x)['uniform']);self.assertTrue(np.array_equal(x,before))
    def test_shape_invalid(self):
        with self.assertRaisesRegex(ValueError,'SHAPE'):rgb.validate(np.zeros((224,224,4),dtype=np.uint8))
    def test_dtype_invalid(self):
        with self.assertRaisesRegex(ValueError,'DTYPE'):rgb.validate(np.zeros((224,224,3),dtype=np.float32))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));Path(__file__).with_name('CPU_TEST_RESULT.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),failures=len(r.failures),errors=len(r.errors)),indent=2)+'\n');raise SystemExit(not r.wasSuccessful())
