import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test_short_worker',HERE/'worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Tests(unittest.TestCase):
    def test_adaptation(self):
        source=worker.source_adapter((HERE.parent/'assembly_v1/worker.py').read_text())
        compile(source,'adapted','exec')
        self.assertIn("split='candidate_fit_pool'",source)
        self.assertNotIn("split='FIT'",source)
        self.assertEqual(source.count("save(folder/'CERTIFICATE.json',certificate)"),1)
        self.assertLess(source.index("save(folder/'CERTIFICATE.json',certificate)"),source.index("export_family(folder/"))
        self.assertIn('certificate=factory.replay_seeds(candidate)',source)
    def test_source_drift_rejected(self):
        with self.assertRaises(AssertionError):worker.source_adapter('drift')

if __name__=='__main__':unittest.main()
