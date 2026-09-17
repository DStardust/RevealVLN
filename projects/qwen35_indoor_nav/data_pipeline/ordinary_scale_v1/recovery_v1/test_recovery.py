import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recovery_audit',HERE/'audit.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

class RecoveryTest(unittest.TestCase):
    def test_bad_route_excluded_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);old=a.HERE;a.HERE=root
            fn=a.strict.audit_route
            def fake(root,j):
                if j['job_id']=='bad':raise AssertionError('TURN_DISPLACEMENT')
                return [dict(job_id=j['job_id'],decisions=3)],set(),3
            a.strict.audit_route=fake
            try:
                result=a.commit_shard(root,0,[dict(job_id=k,source='fixture',scene_id='s') for k in ('good','bad')])
                self.assertEqual(result['certified_routes'],1)
                self.assertEqual(result['quarantined_routes'],1)
                self.assertEqual(result['instruction_conditioned_decisions'],3)
                rows=[json.loads(x) for x in (root/'shards/shard_0000.jsonl').read_text().splitlines()]
                self.assertEqual([r['job_id'] for r in rows],['good'])
                with self.assertRaises(AssertionError):a.commit_shard(root,0,[])
            finally:a.HERE=old;a.strict.audit_route=fn

    def test_infrastructure_error_is_fatal(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d);old=a.HERE;a.HERE=root;fn=a.strict.audit_route
            def fake(root,j):raise OSError('disk')
            a.strict.audit_route=fake
            try:
                with self.assertRaises(OSError):a.commit_shard(root,0,[dict(job_id='x')])
                self.assertFalse((root/'shards/shard_0000.jsonl').exists())
            finally:a.HERE=old;a.strict.audit_route=fn

if __name__=='__main__':unittest.main()

