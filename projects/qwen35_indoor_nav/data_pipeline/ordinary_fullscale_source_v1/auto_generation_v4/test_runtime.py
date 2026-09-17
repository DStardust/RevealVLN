import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import run
import transport
class Tests(unittest.TestCase):
    def test_contexts_ast_exact_original(self):
        def contexts(code):return next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='contexts')
        self.assertEqual(ast.dump(contexts(transport.original('run.py'))),ast.dump(contexts(transport.run_source())))
    def test_only_active_call_replaced(self):
        code=transport.run_source()
        self.assertEqual(code.count('active_accounting(snapshot,proc.pid,out,shard,snapshot_elapsed)'),1)
        self.assertIn('contexts(idle,restorable=True)',code)
        self.assertIn('contexts(snapshot,restorable=True);return snapshot',code)
        self.assertNotIn('contexts(snapshot,proc.pid)',code)
    def test_disjoint_source_and_counts(self):
        jobs=transport.rescue_jobs();self.assertEqual([len(jobs[s]) for s in (2,4)],[364,998])
        self.assertEqual(sum(len(j['instruction_alias_episodes']) for rows in jobs.values() for j in rows),4085)
        for s in (2,4):
            self.assertEqual(c.shard_root(s),HERE.parent/'auto_generation_v4_production'/f'shard_{s:04d}')
            self.assertFalse(c.shard_root(s).is_relative_to(HERE))
    def test_explicit_amendment_not_false_original_guard_claim(self):
        code=transport.prepare_source()
        self.assertIn('gpu_accounting_guard_amended=True',code)
        self.assertIn('disk_census_unchanged=True',code)
        self.assertNotIn('original_safe_size_and_resource_guards_unchanged=True',code)
        self.assertNotIn('guard_amended=False',(HERE/'queue_job.py').read_text())
    def test_actual_durable_journal_links_raw_sample(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder);snapshot=dict(memory_mib=733,processes={1:246,2:246,3:245})
            before=copy.deepcopy(snapshot)
            result=run.active_accounting(snapshot,3,root,2,19.5)
            rows=[json.loads(x) for x in (root/'GPU_ACCOUNTING_DECISIONS.jsonl').read_text().splitlines()]
            self.assertEqual(snapshot,before);self.assertTrue(result['amended_acceptance'])
            self.assertEqual(rows[-1]['original_sample_elapsed'],19.5)
            self.assertEqual(rows[-1]['original_sample_file'],'GPU_SNAPSHOTS.jsonl')
            self.assertEqual(rows[-1]['conservative_upper_mib'],245)
            self.assertFalse(rows[-1]['original_guard_passed'])
    def test_fsync_failure_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder,mock.patch.object(run.os,'fsync',side_effect=OSError('fixture io')):
            with self.assertRaises(OSError):run.active_accounting(dict(memory_mib=733,processes={3:737}),3,Path(folder),2,1)
    def test_original_disk_guard_unchanged(self):
        code=transport.run_source()
        self.assertIn("value=safe_size.measure(c.shard_root(shard))",code)
        self.assertIn("safe_size.measure(HERE)['apparent_bytes_conservative']<3*1024**3",code)
        self.assertIn("<48*1024**3",code)
    def test_full_compile_and_budget(self):
        compile(transport.run_source(),'run','exec');compile(transport.prepare_source(),'prepare','exec')
        self.assertIn("['prior_wall_seconds']",transport.run_source())
        self.assertNotIn("'sleep 24000')",transport.run_source())
if __name__=='__main__':unittest.main(verbosity=2)
