import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('tested_raw_writer_v2',HERE/'audit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_exact_only_writer_changed(self):
        source=m.V1.read_text();new=m.adapted_source(source)
        self.assertEqual(new.replace("_write_report(output/'REPORT.json',result)","old.save(output/'REPORT.json',result)"),source)
    def test_local_exclusive_writer_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d)/'REPORT.json';m._write_report(p,{'x':1})
            self.assertEqual(json.loads(p.read_text()),{'x':1})
            with self.assertRaises(FileExistsError):m._write_report(p,{'x':2})
    def test_other_directory_denied(self):
        with self.assertRaisesRegex(ValueError,'OUTPUT_SCOPE'):m._write_report(HERE.parent/'REPORT.json',{})
    def test_wrong_filename_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            with self.assertRaises(ValueError):m._write_report(Path(d)/'OTHER.json',{})
    def test_symlink_denied(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);(root/'actual').mkdir();(root/'link').symlink_to(root/'actual',target_is_directory=True)
            with self.assertRaises(ValueError):m._write_report(root/'link/REPORT.json',{})
    def test_old_source_change_denied(self):
        with self.assertRaisesRegex(ValueError,'SEALED_V1_CHANGED'):m.adapted_source(m.V1.read_text()+'\n')
    def test_old_20_cases_use_new_loaded_functions(self):
        path=HERE.parent/'telemetry_consistency_audit_v1/test_audit.py'
        s=importlib.util.spec_from_file_location('reuse_raw20_v2',path);tests=importlib.util.module_from_spec(s);s.loader.exec_module(tests)
        tests.a=m;result=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(tests.Tests).run(result)
        self.assertEqual(result.testsRun,20);self.assertEqual(result.errors,[]);self.assertEqual(result.failures,[])
if __name__=='__main__':unittest.main()
