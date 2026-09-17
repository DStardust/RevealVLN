import importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test_batch_shared',HERE/'shared.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_worker_preserves_verifiers(self):
        source=m.worker_source((m.WF/'assembly_v1/worker.py').read_text())
        compile(source,'worker','exec')
        self.assertIn("balance=row['balance']",source)
        self.assertIn("split='candidate_fit_pool'",source)
        self.assertIn('factory.replay_seeds(candidate)',source)
        self.assertIn('loader.validate_supervision_contract()',source)
        self.assertEqual(source.count("save(folder/'CERTIFICATE.json',certificate)"),1)
    def test_gpu_supervisor(self):
        for gpu in (1,2):
            source=m.supervisor_source((m.RUNTIME/'compact_loop_v2/run.py').read_text(),gpu)
            compile(source,'supervisor','exec')
            self.assertIn(f"'nvidia-smi','-i','{gpu}'",source)
            self.assertIn("sample['elapsed'] < 3900",source)
            self.assertIn("sample['disk_bytes'] < 7*1024**3",source)
            self.assertIn('os.killpg(proc.pid,signal.SIGTERM)',source)
        with self.assertRaises(AssertionError):m.supervisor_source('',0)
if __name__=='__main__':unittest.main()
