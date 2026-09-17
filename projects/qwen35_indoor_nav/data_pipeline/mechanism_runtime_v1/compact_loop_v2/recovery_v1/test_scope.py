import importlib.util
from pathlib import Path
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
def load():
    spec=importlib.util.spec_from_file_location('scope_test_worker',HERE/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class Tests(unittest.TestCase):
    def test_exact_new_store_scope_and_actual_raw_commit(self):
        cls=load().scoped_store()
        with tempfile.TemporaryDirectory(prefix='cpu_store_',dir=HERE) as root:
            with cls(Path(root)/'content',1024**2) as store:
                item=store.put_raw(bytes([1,2,3]),'rgb',[1,1,3],'uint8')
                self.assertEqual(item['pixel_sha256'],'039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81')
            self.assertTrue(store.final_audit['audit_pass'])
    def test_outside_recovery_scope_rejected(self):
        cls=load().scoped_store()
        with self.assertRaises(ValueError):cls(HERE.parent/'forbidden_store',1024**2)
    def test_worker_full_source_import_and_bind(self):
        module=load(); adapter=module.load('scope_adapter',HERE.parent/'worker.py')
        self.assertIn('rank_reachable_positions',adapter.adapted_source())
if __name__=='__main__':unittest.main()
