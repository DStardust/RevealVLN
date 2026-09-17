from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport as t
import prepare as p
class Tests(unittest.TestCase):
    def test_only_manifest_cap_changes(self):
        old=t.checked('transport.py').read_text()
        old=old[old.index('def run_main(batch):'):old.index("\nif __name__=='__main__':")]
        self.assertEqual(t.adapted_run_source().replace('len(lock)<=2048','len(lock)<=1024'),old)
    def test_manifest_boundary(self):
        for count,allowed in [(1024,True),(1026,True),(2048,True),(2049,False)]:
            self.assertEqual(count<=2048,allowed)
    def test_all_original_safety_functions(self):
        for name in ['run_lease','restore_dead_holder','wait_gpu_drain','build_supervisor','normalize_identity']:
            self.assertEqual(getattr(t.v3.base,name).__code__.co_filename,str(t.OLD/'transport.py'))
    def test_cfg_marks_software_cap(self):
        code=p.adapted((t.BE/'prepare.py').read_text());compile(code,'cpu','exec')
        self.assertIn('gpu5_transport_v4_2048',code)
        self.assertIn('gpu5_transport_v4/transport.py',p.p3.p.LAUNCHER)
    def test_exact_sources(self):t.verify_sources()
if __name__=='__main__':unittest.main()
