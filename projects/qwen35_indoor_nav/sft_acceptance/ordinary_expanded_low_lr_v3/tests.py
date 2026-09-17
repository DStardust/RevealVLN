"""Resume and unchanged-algorithm contract checks, CPU-only."""
import ast
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location('continued_test_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('reuse');d=load('data');c=load('control')


class ContinueTests(unittest.TestCase):
    def test_sources_compile(self):
        for name in r.HASHES:ast.parse(r.source(name))

    def test_same_math_model_and_control(self):
        for name in ('model.py','data.py','control.py','launcher.py','lease_run.py'):
            self.assertEqual(r.source(name),r.parent.source(name))

    def test_train_only_id_and_index_path(self):
        a=r.source('train_filestore.py')
        a=a.replace("'Q35N_ORDINARY_EXPANDED_LOW_LR_V3'","'Q35N_ORDINARY_EXPANDED_V1'")
        a=a.replace("HERE.parent / 'ordinary_expanded_v1/SAMPLE_INDEX.jsonl'","HERE / 'SAMPLE_INDEX.jsonl'")
        self.assertEqual(a,r.parent.source('train_filestore.py'))

    def test_optimizer_cursor_rng_restored(self):
        text=r.source('train_filestore.py')
        for anchor in ("opt.load_state_dict(copy.deepcopy(state['optimizer']))","cursor.update(state['cursor'])","torch.set_rng_state(state['torch_rng'])","random.setstate(state['python_rng'])"):
            self.assertIn(anchor,text)
        self.assertIn("shard = plans[epoch][rank][position:]",text)
        self.assertIn("cosine_warmup(cursor['updates'], total_updates",text)

    def test_no_auto_retry(self):
        text=r.source('supervise_filestore.py')
        self.assertIn('range(1, 2)',text);self.assertIn('max_attempts=1',text)

    def test_data_wrapper_byte_exact(self):
        self.assertEqual((HERE/'data.py').read_bytes(),(r.OLD/'data.py').read_bytes())

    def test_next_slice_no_duplicates(self):
        samples=[dict(est=170+i%100) for i in range(2500)]
        plan=d.plan_epoch_batches(samples,6144,1209,0,3)
        first={i for row in plan for batch in row[:5] for i in batch}
        second={i for row in plan for batch in row[5:10] for i in batch}
        self.assertTrue(first.isdisjoint(second))

    def test_rank_counts_not_copied_from_rank_zero(self):
        plans=[[[[0,1],[2]],[[3],[4,5]],[[6,7,8],[9]]]]
        cursor=dict(epoch=0,position=1,updates=1,decisions=2)
        self.assertEqual([c.processed_by_rank(plans,cursor,r) for r in range(3)],[2,1,3])

    def test_old_count_not_immediate_budget_stop(self):
        self.assertEqual(c.stop_mask([],10,100,4000,8000,340712,700712),0)
        self.assertEqual(c.stop_mask([],10,100,8000,8000,680000,700712),4)

    def test_reject_invalid_cursor(self):
        with self.assertRaises(ValueError):c.processed_by_rank([[[[1]]]],dict(epoch=0,position=1,updates=0),0)


if __name__=='__main__':unittest.main()
