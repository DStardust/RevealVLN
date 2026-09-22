import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
s=importlib.util.spec_from_file_location('disk',Path(__file__).with_name('disk.py'));disk=importlib.util.module_from_spec(s);s.loader.exec_module(disk)
class Tests(unittest.TestCase):
    def test_static_and_hardlink(self):
        import os
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a').write_bytes(b'x'*53);os.link(p/'a',p/'b')
            self.assertEqual(disk.measure(p)['bytes'],p.lstat().st_size+53)
    def test_atomic_writer_disappearance(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'rename.tmp').write_bytes(b'x');original=Path.lstat
            def transient(v,*a,**k):
                if v.name=='rename.tmp':raise FileNotFoundError('writer renamed temporary file')
                return original(v,*a,**k)
            with patch.object(Path,'lstat',transient):r=disk.measure(p)
            self.assertEqual(r['vanished_during_scan'],1);self.assertEqual(r['bytes'],p.lstat().st_size)
    def test_permission_errors_are_not_hidden(self):
        with patch.object(Path,'lstat',side_effect=PermissionError):
            with self.assertRaises(PermissionError):disk.measure(Path('/denied'))
    def test_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'link').symlink_to('/proc');r=disk.measure(p)
            self.assertEqual(r['files'],1);self.assertEqual(r['bytes'],p.lstat().st_size+(p/'link').lstat().st_size)
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('CPU_TEST_RESULT.json').write_text(json.dumps(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful()),indent=2)+'\n')
    raise SystemExit(not r.wasSuccessful())
