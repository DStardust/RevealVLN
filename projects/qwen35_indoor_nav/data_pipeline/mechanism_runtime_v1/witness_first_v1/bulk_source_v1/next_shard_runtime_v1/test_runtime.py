import importlib.util
from pathlib import Path
import unittest
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('next_scout_test_runtime', HERE/'runtime.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

class Tests(unittest.TestCase):
    def test_sources_compile(self):
        for fn in (r.common_source,r.worker_source,r.supervisor_source):
            compile(fn(), '<cpu>', 'exec')
    def test_no_algorithm_change_in_worker(self):
        source = r.worker_source().replace(repr(str(HERE)), repr(str(r.OLD_ROOT)))
        source = source.replace("HabitatBackend(candidate['scene_glb'],2,candidate['roles']", "HabitatBackend(candidate['scene_glb'],1,candidate['roles']")
        self.assertEqual(source,r.a.worker_source(1))
    def test_guard_unchanged(self):
        old = r.a.checked(r.a.RUNTIME/'compact_loop_v2/run.py')
        guard = old[old.index('def check_gpu('):old.index('def disk_size(')]
        self.assertIn(guard,r.supervisor_source())
        self.assertIn("sample['elapsed'] < 4500",r.supervisor_source())
        self.assertIn("sample['disk_bytes'] < 8*1024**3",r.supervisor_source())
    def test_correct_gpu_and_worker(self):
        s = r.supervisor_source()
        self.assertIn("'nvidia-smi','-i','2'",s)
        self.assertIn(r.UUID,s)
        self.assertIn(repr(str(HERE/'worker.py')),s)
        self.assertNotIn("'--shard','0'",s)
    def test_private_content_store_scope(self):
        m = r.module('cpu_scope_next',r.common_source(),r.a.SCOUT/'common.py')
        self.assertEqual(m.HERE,HERE)
        cls = m.store_class()
        self.assertEqual(cls.__init__.__globals__['FEEDBACK_ROOT'],HERE)
    def test_common_requires_runtime_approval(self):
        m = r.module('cpu_failclosed_next',r.common_source(),r.a.SCOUT/'common.py')
        with self.assertRaises(RuntimeError):m.runtime_config()
    def test_three_different_fit_houses(self):
        rows = r.p.read(r.BULK/'INVENTORY_ALL_37.json')['rows']
        ids = {x['house_id'] for x in rows[3:6]}
        self.assertEqual(len(ids),3)
        self.assertFalse(ids & {x['house_id'] for x in rows[:3]})
        self.assertTrue(ids <= set(r.p.read(r.p.SPLIT)['FIT']))

if __name__ == '__main__':unittest.main()
