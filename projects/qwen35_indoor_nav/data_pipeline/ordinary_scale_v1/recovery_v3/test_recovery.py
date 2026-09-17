import errno
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('safe_size_test', HERE/'safe_size.py')
size = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size)

class SafeSizeTests(unittest.TestCase):
    def fake_stat_test(self, filename, error, tolerated):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            path = root / 'content' / filename
            path.parent.mkdir()
            path.touch()
            def fake(p):
                if Path(p) == path:
                    raise error
                return os.lstat(p)
            if tolerated:
                result = size.measure(root, fake)
                self.assertEqual(result['tolerated_atomic_disappearances'], [str(path.relative_to(root))])
                self.assertGreaterEqual(result['apparent_bytes_conservative'], 1024**2)
            else:
                with self.assertRaises(type(error)):
                    size.measure(root, fake)

    def test_png_promotion_race_allowed(self):
        self.fake_stat_test('a'*64+'.png.tmp', FileNotFoundError(errno.ENOENT, 'gone'), True)

    def test_missing_normal_artifact_fatal(self):
        self.fake_stat_test('a'*64+'.png', FileNotFoundError(errno.ENOENT, 'gone'), False)

    def test_unknown_tmp_fatal(self):
        self.fake_stat_test('unknown.tmp', FileNotFoundError(errno.ENOENT, 'gone'), False)

    def test_permission_fatal_even_for_temporary(self):
        self.fake_stat_test('a'*64+'.png.tmp', PermissionError(errno.EACCES, 'denied'), False)

    def test_io_fatal_even_for_temporary(self):
        self.fake_stat_test('a'*64+'.png.tmp', OSError(errno.EIO, 'io'), False)

    def test_root_outside_scope_fatal(self):
        with self.assertRaises(ValueError):
            size.measure(Path('/'))

    def test_symlink_fatal(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            (root/'link').symlink_to(root)
            with self.assertRaises(ValueError):
                size.measure(root)

if __name__ == '__main__':
    unittest.main()
