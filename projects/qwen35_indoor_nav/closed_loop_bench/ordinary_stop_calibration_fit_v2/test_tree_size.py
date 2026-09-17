import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scan_test',HERE/'tree_size.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class ScanTests(unittest.TestCase):
    def test_normal(self):
        with tempfile.TemporaryDirectory(prefix='scan_test_',dir=HERE) as tmp:
            p=Path(tmp);(p/'a').write_bytes(b'abc');(p/'sub').mkdir();(p/'sub/b').write_bytes(b'12345')
            self.assertEqual(m.tree_size(p),(8,0))
    def test_directory_renamed_during_scan(self):
        with tempfile.TemporaryDirectory(prefix='scan_test_',dir=HERE) as tmp:
            p=Path(tmp);(p/'stable').write_bytes(b'abc');(p/'temp').mkdir();original=m.os.scandir
            def scan(path):
                if Path(path)==p/'temp':raise FileNotFoundError(path)
                return original(path)
            with mock.patch.object(m.os,'scandir',side_effect=scan):self.assertEqual(m.tree_size(p),(3,1))
    def test_permissions_not_suppressed(self):
        with tempfile.TemporaryDirectory(prefix='scan_test_',dir=HERE) as tmp:
            with mock.patch.object(m.os,'scandir',side_effect=PermissionError):
                with self.assertRaises(PermissionError):m.tree_size(tmp)
    def test_missing_root_not_suppressed(self):
        with self.assertRaises(FileNotFoundError):m.tree_size(HERE/'nonexistent_test_root')
    def test_symlink_not_followed(self):
        with tempfile.TemporaryDirectory(prefix='scan_test_',dir=HERE) as tmp:
            p=Path(tmp);(p/'a').write_bytes(b'x');(p/'link').symlink_to(p/'a')
            with self.assertRaises(RuntimeError):m.tree_size(p)


if __name__=='__main__':unittest.main()
