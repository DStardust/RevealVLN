import ast
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('full_epoch_source_tests',HERE/'reuse.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


class FullEpochTests(unittest.TestCase):
    def test_unchanged_algorithm(self):
        for name in ('model.py','data.py','control.py','launcher.py','lease_run.py'):
            self.assertEqual(r.source(name),r.parent.source(name))
    def test_only_training_id_changed(self):
        text=r.source('train_filestore.py').replace('Q35N_ORDINARY_EXPANDED_FULL_EPOCH_V4','Q35N_ORDINARY_EXPANDED_CONTINUE_V2')
        self.assertEqual(text,r.parent.source('train_filestore.py'))
    def test_transport_only_cache_names(self):
        text=r.source('supervise_filestore.py').replace('ordinary_expanded_full_epoch_v4','ordinary_expanded_continue_v2')
        self.assertEqual(text,r.parent.source('supervise_filestore.py'))
    def test_syntax(self):
        for name in ('model.py','data.py','control.py','launcher.py','lease_run.py','train_filestore.py','supervise_filestore.py'):
            ast.parse(r.source(name))
    def test_epoch_boundary_not_max_updates_stop(self):
        text=r.source('train_filestore.py')
        self.assertIn('new_cursor = data.advance_epoch_boundary',text)
        self.assertIn("status = 'EPOCHS_COMPLETED'",text)
    def test_resume_optimizer_and_plan(self):
        text=r.source('train_filestore.py')
        self.assertIn("opt.load_state_dict(copy.deepcopy(state['optimizer']))",text)
        self.assertIn('control.processed_by_rank(plans, cursor, rank)',text)


if __name__=='__main__':unittest.main()
