import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import transport
import prepare
import run
import sentinel_gate
import telemetry

class Tests(unittest.TestCase):
    def test_transports_compile_and_roots(self):
        for s in range(21):
            code=c.worker_source(s);compile(code,'worker','exec')
            self.assertIn('cfg.gpu_device_id=6;',code)
            self.assertEqual(c.shard_root(s).parent,transport.DATA)
            self.assertFalse(c.shard_root(s).is_relative_to(HERE))
        for s in (-1,21,True):
            with self.assertRaises(AssertionError):c.shard_root(s)
        for f in (transport.common_source,transport.run_source,transport.merge_source):compile(f(),'adapted','exec')

    def test_frozen_source_partition_and_sentinel(self):
        rows=c.read(transport.MANIFEST/'JOBS.json');plan=c.read(transport.MANIFEST/'PLAN.json')
        shards=[c.read(transport.MANIFEST/r['jobs_path']) for r in plan['shards']]
        self.assertEqual([j for jobs in shards for j in jobs],rows)
        self.assertEqual(len(rows),20000)
        self.assertEqual(len({j['physical_source_route_sha256'] for j in rows}),20000)
        self.assertEqual(len({j['scene_id'] for j in rows}),50)
        self.assertEqual([j['official_gt_actions'] for j in shards[0]],[36,69,52])
        self.assertEqual(len({j['scene_id'] for j in shards[0]}),3)

    def test_base_contexts_readiness_and_restore_ast_unchanged(self):
        def fn(source,name):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))
        for name in ('contexts','restore_holder','verify_identity','drain_context','parse_gpu'):
            self.assertEqual(fn(transport.original('run.py'),name),fn(transport.run_source(),name))
        source=transport.run_source()
        self.assertNotIn("'sleep 24000'",source)
        self.assertNotIn("'respawn-pane','-k'",source)
        self.assertIn('NEW_ENVDROP_LANE_NO_AUTOMATIC_RETRY',source)
        self.assertIn('FULL_BATCH_REQUIRED_NO_PARTIAL_PROMOTION',transport.merge_source())

    def test_tail_budget_never_starts_short_phase(self):
        with mock.patch.object(c,'save') as save:
            run.require_phase_budget(82800,78480,4200,HERE,20)
            save.assert_not_called()
            with self.assertRaisesRegex(AssertionError,'INSUFFICIENT_FULL_PHASE'):
                run.require_phase_budget(82800,78480.001,4200,HERE,20)
            self.assertEqual(save.call_args.args[1]['status'],'UNATTEMPTED_BUDGET_CENSOR')

    def test_no_approval_no_gpu(self):
        with mock.patch.object(c,'approved',side_effect=AssertionError('approval')),mock.patch.object(run,'gpu_snapshot') as gpu:
            with self.assertRaisesRegex(AssertionError,'approval'):run.execute(6)
            gpu.assert_not_called()

    def test_active_actual_733_737_preserves_raw(self):
        snap=dict(memory_mib=733,processes={1:246,2:246,3:245});old=copy.deepcopy(snap);logs=[]
        result=telemetry.active_assessment(snap,3,run.contexts,logs.append)
        self.assertEqual(snap,old);self.assertEqual(result['conservative_upper_mib'],245)
        self.assertFalse(result['original_guard_passed']);self.assertTrue(result['amended_acceptance'])
        with self.assertRaisesRegex(AssertionError,'GPU_MEMORY_ACCOUNTING'):run.contexts(snap,restorable=True)

    def test_active_negative_unknown_and_overage_reject(self):
        for total,processes in [(4095,{1:246,3:4000}),(500,{1:769,3:1}),(-1,{3:1}),(float('nan'),{3:1})]:
            with self.assertRaises(AssertionError):telemetry.active_assessment(dict(memory_mib=total,processes=processes),3,run.contexts,lambda x:None)
        with self.assertRaisesRegex(AssertionError,'GPU_MEMORY_ACCOUNTING'):
            telemetry.active_assessment(dict(memory_mib=733,processes={1:246,2:246,4:245}),3,run.contexts,lambda x:None)

    def fixture(self,root):
        (root/'shards').mkdir();jobs=[];ledger=[];rows=[]
        for i in range(3):
            j=dict(job_id=str(i),scene_id='house'+str(i),source='ENVDROP_OFFICIAL_CE_TRAIN',source_sha256='a'*64,
                physical_source_route_sha256=str(i)*64,instruction_alias_episodes=[{'episode_id':i}])
            jobs.append(j);ledger.append(dict(job_id=str(i),status='CERTIFIED'))
            for name in (f'policy_{i}.json',f'sup_{i}.json'):(root/name).touch()
            rows.append(dict(job_id=str(i),source=j['source'],source_sha256=j['source_sha256'],split='FIT',scene_group=j['scene_id'],
                physical_source_route_sha256=j['physical_source_route_sha256'],policy_file=f'policy_{i}.json',supervision_file=f'sup_{i}.json',decisions=5))
        for path,value in [(root/'JOBS.json',jobs),(root/'GENERATION_COMPLETE.json',{'all_jobs_terminal':True}),(root/'quarantine.json',[])]:c.save(path,value)
        (root/'LEDGER.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
        idx=root/'shards/shard_0000.jsonl';idx.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c.save(root/'shards/shard_0000.audit.json',dict(integrity_pass=True,index_sha256=c.sha(idx),quarantine_manifest='quarantine.json',
            certified_routes=3,quarantined_routes=0,instruction_records=3,instruction_conditioned_decisions=15))
        strict=types.SimpleNamespace(audit_route=lambda root,j:([rows[int(j['job_id'])]],set(),5))
        return jobs,strict

    def test_sentinel_three_pass_only(self):
        with tempfile.TemporaryDirectory(dir=HERE) as path:
            root=Path(path);jobs,strict=self.fixture(root)
            original=c.sha
            with mock.patch.object(c,'sha',side_effect=lambda p:'fake_lock' if p==HERE/'INPUT_LOCK.json' else original(p)):
                value=sentinel_gate.inspect(root,jobs,strict)
            self.assertTrue(value['strict_pass']);self.assertEqual(value['strict_routes'],3)
            self.assertFalse(value['scientific_pass'])

    def test_sentinel_missing_rejected_or_quarantined_fails(self):
        for mutation in ('missing','reject','quarantine','fresh_mismatch'):
            with tempfile.TemporaryDirectory(dir=HERE) as path:
                root=Path(path);jobs,strict=self.fixture(root)
                if mutation in ('missing','reject'):
                    rows=sentinel_gate.merge.rows(root/'LEDGER.jsonl')
                    if mutation=='missing':rows.pop()
                    else:rows[0]['status']='REJECTED'
                    (root/'LEDGER.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                elif mutation=='quarantine':(root/'quarantine.json').write_text('[{"job_id":"0"}]')
                else:strict.audit_route=lambda root,j:([],set(),0)
                with self.assertRaises(AssertionError):sentinel_gate.inspect(root,jobs,strict)

    def test_sentinel_receipt_rebind_reject(self):
        value=dict(strict_pass=True,strict_routes=3,source_jobs_sha256='old',input_lock_sha256='x',input_hashes={})
        with mock.patch.object(c,'read',return_value=value),mock.patch.object(c,'sha',return_value='changed'):
            with self.assertRaises(AssertionError):run.require_sentinel_receipt(HERE)

    def test_sentinel_runner_fails_and_never_changes_jobs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as path,mock.patch.object(run.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)),mock.patch.object(run,'require_sentinel_receipt') as receipt:
            with self.assertRaisesRegex(AssertionError,'NO_REPLACEMENT'):run.run_sentinel_gate(Path(path),{})
            receipt.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
