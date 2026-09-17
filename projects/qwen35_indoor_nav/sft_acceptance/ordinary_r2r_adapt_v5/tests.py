import ast
import importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('r2r_tests_reuse',HERE/'reuse.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class Tests(unittest.TestCase):
    def test_sources_parse(self):
        for name in r.HASHES:ast.parse(r.source(name))
    def test_model_and_loss_unchanged(self):
        self.assertEqual(r.source('model.py'),r.parent.source('model.py'))
        t=r.source('train_filestore.py')
        self.assertIn("loss = (ce * batch['weights']).sum() * world / weight_sum_global",t)
        self.assertIn("total_updates = protocol['lr_reference_total_updates']",t)
        self.assertIn("warmup = max(1, int(0.03 * total_updates))",t)
    def test_only_two_training_edits(self):
        text=r.parent.source('train_filestore.py')
        text=text.replace("'Q35N_ORDINARY_EXPANDED_CONTINUE_V2'","'Q35N_ORDINARY_R2R_ADAPT_V5'")
        text=text.replace("total_updates = sum(len(p[rank]) for p in plans)","total_updates = protocol['lr_reference_total_updates']  # Preserve original LR clock, not new sampler length.")
        self.assertEqual(r.source('train_filestore.py'),text)
    def test_one_attempt(self):
        self.assertIn('range(1, 2)',r.source('supervise_filestore.py'))
        self.assertNotIn('range(1, 4)',r.source('supervise_filestore.py'))
    def test_inputs_remain_causal(self):
        t=r.source('model.py')
        self.assertIn("past_key_values=None, use_cache=False",t)
        self.assertNotIn('stop_logit_bias',t)
if __name__=='__main__':unittest.main()
