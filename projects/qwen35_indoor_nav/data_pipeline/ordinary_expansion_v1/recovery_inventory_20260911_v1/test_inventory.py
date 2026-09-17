import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('inventory',HERE/'inventory.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_partition(self):
        jobs=[dict(job_id=x) for x in 'abc']
        p=m.partition(jobs,[dict(job_id='a')],{'a','b'})
        self.assertEqual(p,dict(terminal=[jobs[0]],partial=[jobs[1]],unattempted=[jobs[2]]))
    def test_unknown_directory_refused(self):
        with self.assertRaises(AssertionError):m.partition([dict(job_id='a')],[],{'b'})
    def test_missing_terminal_directory_refused(self):
        with self.assertRaises(AssertionError):m.partition([dict(job_id='a')],[dict(job_id='a')],set())
    def test_duplicate_jobs_refused(self):
        with self.assertRaises(AssertionError):m.partition([dict(job_id='a')]*2,[],set())
    def test_empty_ledger(self):self.assertEqual(m.ledger(b'')[0],[])
    def test_partial_tail_not_a_label(self):
        rows,tail=m.ledger(b'{"job_id":"a"}\n{"job_id":"b"}')
        self.assertEqual(rows,[dict(job_id='a')]);self.assertGreater(tail['bytes'],0)
    def test_duplicate_ledger_refused(self):
        with self.assertRaises(AssertionError):m.ledger(b'{"job_id":"a"}\n'*2)
    def test_broken_interior_ledger_refused(self):
        with self.assertRaises(json.JSONDecodeError):m.ledger(b'{broken\n')
    def test_duplicate_alias_records_supported_only_in_index(self):
        self.assertEqual(len(m.ledger(b'{"job_id":"a"}\n'*2,False)[0]),2)
    def fixture(self):
        job=dict(job_id='a',source='source',source_sha256='s',scene_id='house',physical_source_route_sha256='p',instruction_alias_episodes=[dict(episode_id=1)])
        row=dict(job_id='a',source='source',source_sha256='s',scene_group='house',physical_source_route_sha256='p',split='FIT',policy_file='routes/a/policy_1.json')
        return job,row
    def test_accept_and_duplicate_physical(self):
        j,r=self.fixture();p=set();a=set();m.accept(j,[r],p,a)
        self.assertEqual(p,{'p'});self.assertEqual(a,{('source','1')})
        with self.assertRaises(AssertionError):m.accept(j,[r],p,a)
    def test_duplicate_alias(self):
        j,r=self.fixture()
        with self.assertRaises(AssertionError):m.accept(j,[r],set(),{('source','1')})
    def test_wrong_scene_and_split_refused(self):
        j,r=self.fixture()
        for bad in (dict(r,scene_group='other'),dict(r,split='DEV')):
            with self.assertRaises(AssertionError):m.accept(j,[bad],set(),set())
    def test_wrong_alias_file_refused(self):
        j,r=self.fixture()
        with self.assertRaises(AssertionError):m.accept(j,[dict(r,policy_file='policy_2.json')],set(),set())
    def test_strict_source_unchanged(self):self.assertEqual(m.sha(m.STRICT),m.STRICT_SHA)


if __name__=='__main__':unittest.main()
