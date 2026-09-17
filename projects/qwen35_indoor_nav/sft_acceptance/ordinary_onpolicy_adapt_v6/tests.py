import ast,importlib.util,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v6_source_tests',HERE/'reuse.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class Tests(unittest.TestCase):
    def test_sources(self):
        for name in r.HASHES:ast.parse(r.source(name))
    def test_same_model(self):self.assertEqual(r.source('model.py'),r.parent.source('model.py'))
    def test_offset_and_same_loss(self):
        t=r.source('train_filestore.py')
        self.assertIn("cursor['updates'] + protocol['optimizer_step_offset']",t)
        self.assertIn("loss = (ce * batch['weights']).sum() * world / weight_sum_global",t)
        self.assertIn("HERE / 'SAMPLE_INDEX.jsonl'",t)
    def test_one_attempt(self):self.assertIn('range(1, 2)',r.source('supervise_filestore.py'))
    def test_whitelist(self):
        t=(HERE/'data.py').read_text()
        self.assertIn("set(value)=={'record_id','instruction','rgb_sha256','executed_actions'}",t)
        self.assertNotIn("return dict(goal",t)
if __name__=='__main__':unittest.main()
