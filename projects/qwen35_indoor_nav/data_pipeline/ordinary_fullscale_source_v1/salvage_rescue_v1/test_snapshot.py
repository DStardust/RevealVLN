import json
from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import snapshot as s
import gate


class Tests(unittest.TestCase):
    def test_three_way_partition_no_retry(self):
        jobs=[dict(job_id=x) for x in 'abcd']
        out=s.partition(jobs,[dict(job_id='a',status='FAILED'),dict(job_id='b',status='CERTIFIED')],{'a','b','c'})
        self.assertEqual([x['job_id'] for x in out['unattempted']],['d'])
        self.assertEqual([x['job_id'] for x in out['interrupted']],['c'])
    def test_terminal_without_dir_rejected(self):
        with self.assertRaisesRegex(AssertionError,'TERMINAL_WITHOUT_DIRECTORY'):
            s.partition([dict(job_id='a')],[dict(job_id='a')],set())
    def test_unknown_dir_rejected(self):
        with self.assertRaisesRegex(AssertionError,'UNREGISTERED_ROUTE'):
            s.partition([dict(job_id='a')],[],{'foreign'})
    def test_ledger_tail_preserved_not_label(self):
        rows,tail=s.ledger_bytes(b'{"job_id":"a"}\n{"job_id":"b"')
        self.assertEqual(rows,[dict(job_id='a')]);self.assertGreater(tail['bytes'],0)
    def test_duplicate_terminal_rejected(self):
        with self.assertRaisesRegex(AssertionError,'DUPLICATE_LEDGER'):
            s.ledger_bytes(b'{"job_id":"a"}\n{"job_id":"a"}\n')
    def test_alias_index_same_job_legal(self):
        rows,tail=s.ledger_bytes(b'{"job_id":"a"}\n{"job_id":"a"}\n',False)
        self.assertEqual(len(rows),2);self.assertEqual(tail['bytes'],0)
    def test_malformed_complete_line_failclosed(self):
        with self.assertRaises(json.JSONDecodeError):s.ledger_bytes(b'{bad}\n')
    def worker_fixture(self):
        return dict(error="AssertionError('SHARD_WALL_BUDGET')",workers=[dict(shard=0,returncode=-15)]),{0:dict(gpu=3,shard=0,pid=123)}
    def test_closed_budget_worker_valid(self):
        r,p=self.worker_fixture();s.validate_workers(r,p,3,lambda pid:False)
    def test_live_worker_refused(self):
        r,p=self.worker_fixture()
        with self.assertRaisesRegex(AssertionError,'OWN_WORKER_STILL_PRESENT'):s.validate_workers(r,p,3,lambda pid:True)
    def test_unknown_budget_error_refused(self):
        r,p=self.worker_fixture();r['error']="AssertionError('EXTERNAL_GPU_PROCESS')"
        with self.assertRaisesRegex(AssertionError,'NOT_REGISTERED_RESOURCE_CENSOR'):s.validate_workers(r,p,3,lambda pid:False)
    def test_gate_exact_resource_scope(self):
        self.assertEqual(gate.RECOVERY,s.FULL.parent/'ordinary_parallel_v1/holder_recovery_v3')
        self.assertIn("assert gpu in (3,4)",gate.source)
        self.assertIn("f'PROCESS_{gpu-3}.json'",gate.source)
        self.assertNotIn("assert result['error'] is None",gate.source)
    def job(self):
        return dict(job_id='a',source='r2r',source_sha256='source',scene_id='house',physical_source_route_sha256='physical',instruction_alias_episodes=[dict(episode_id=1),dict(episode_id=2)])
    def rows(self):
        return [dict(job_id='a',source='r2r',source_sha256='source',scene_group='house',physical_source_route_sha256='physical',split='FIT',policy_file=f'routes/a/policy_{i}.json',supervision_file='routes/a/supervision_only.json') for i in (1,2)]
    def test_complete_alias_validation(self):
        root=s.FULL/'production/shard_0000'
        with mock.patch.object(s,'safe',side_effect=lambda p:p):
            out=s.accepted_rows(root,self.job(),self.rows(),set(),set())
        self.assertEqual(len(out),2);self.assertEqual(out[0]['sourceRoot'],str(root.relative_to(s.ROOT)))
    def test_missing_alias_rejected(self):
        with self.assertRaises(AssertionError):s.accepted_rows(s.FULL,self.job(),self.rows()[:1],set(),set())
    def test_duplicate_physical_rejected(self):
        with self.assertRaisesRegex(AssertionError,'DUPLICATE_PHYSICAL'):s.accepted_rows(s.FULL,self.job(),self.rows(),{'physical'},set())
    def test_duplicate_alias_rejected(self):
        with self.assertRaisesRegex(AssertionError,'DUPLICATE_ALIAS'):s.accepted_rows(s.FULL,self.job(),self.rows(),set(),{('r2r','1')})
    def test_outside_project_rejected(self):
        with self.assertRaisesRegex(AssertionError,'PROJECT_PATH_REQUIRED'):s.safe(Path('/proc'))


if __name__=='__main__':unittest.main(verbosity=2)
