import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load():
    s=importlib.util.spec_from_file_location('test_auto_audit_scope_v2',HERE/'audit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load()
class TestAuditScope(unittest.TestCase):
    def test_exact_differential(self):
        src=a.ORIGINAL.read_text();out=a.adapted_source(src)
        self.assertEqual(out.replace(a.NEW,a.OLD),src)
    def test_source_changed_rejected(self):
        with self.assertRaises(AssertionError):a.adapted_source(a.ORIGINAL.read_text()+'\n')
    def test_no_gpu_on_import(self):
        with patch('subprocess.run',side_effect=AssertionError('NO_PROCESS')),patch('subprocess.check_output',side_effect=AssertionError('NO_QUERY')):load()
    def test_original_save_real_scope_and_readback(self):
        g=a.private.t.load('test_auto_actual_gate3',a.private.GATE)
        root=a.private.GATE.parent/'auto_generation_v1';root.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='TEST_FIXTURE_OUTPUT_SCOPE_',dir=root))
        fixture={'kind':'TEST_FIXTURE','scientific_pass':False,'physical_family':False,'gpu_operations':0}
        path=out/'CPU_SAVE_ONLY.json';g.gate.old.save(path,fixture)
        self.assertEqual(json.loads(path.read_text()),fixture)
        with self.assertRaises(Exception):g.gate.old.save(HERE/'TEST_FIXTURE_FORBIDDEN_SCOPE.json',fixture)
        self.assertFalse((HERE/'TEST_FIXTURE_FORBIDDEN_SCOPE.json').exists())
if __name__=='__main__':unittest.main()
