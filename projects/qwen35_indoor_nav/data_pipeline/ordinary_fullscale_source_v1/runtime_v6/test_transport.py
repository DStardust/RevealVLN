from pathlib import Path
import sys
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport
import common as c


class Tests(unittest.TestCase):
    def test_actual_rescue_subset_and_source_order(self):
        rows=transport.rescue_jobs()
        self.assertEqual({s:len(j) for s,j in rows.items()},{0:279,1:674})
        self.assertEqual(sum(len(j['instruction_alias_episodes']) for js in rows.values() for j in js),2858)
    def test_independent_rescue_roots(self):
        self.assertEqual(c.LANES,{4:(0,1)})
        for shard in (0,1):
            self.assertEqual(c.shard_root(shard),HERE.parent/'rescue_production'/f'shard_{shard:04d}')
            self.assertIn('cfg.gpu_device_id=4;',c.worker_source(shard))
    def test_dead_pane_and_old_lanes_lock(self):
        source=transport.run_source();compile(source,'CPU_run','exec')
        self.assertNotIn("'sleep 24000')",source)
        self.assertNotIn('sleeper_identity=wait_stable_sleeper(identity)',source)
        self.assertIn('old_source_locks=c.claim_original_source_lanes()',source)
    def test_prepare_uses_rescue_only(self):
        source=transport.prepare_source();compile(source,'CPU_prepare','exec')
        self.assertIn('jobs=transport.rescue_jobs()',source)
        self.assertIn('FRESH_RESCUE_PRODUCTION_REQUIRED',source)
        self.assertNotIn("jobs={s:c.read(c.PARALLEL/plan['shards'][s]['jobs_path'])",source)
    def test_budget_boundaries(self):
        auth=dict(cleanup_margin_seconds=120,shard_wall_seconds={'0':4800,'1':18000},lane_wall_seconds={'4':22920},max_wall_seconds_per_gpu_chain=22920)
        self.assertEqual(transport.budget_from_authorization(auth,{4:(0,1)}),({'0':4800,'1':18000},{'4':22920}))
        auth['lane_wall_seconds']['4']=22800
        with self.assertRaisesRegex(AssertionError,'LANE_MUST'):transport.budget_from_authorization(auth,{4:(0,1)})
    def test_worker_strict_and_normal_input_lock(self):
        self.assertIn("c.read(root/'INPUT_LOCK.json')==lock",transport.original('worker.py'))
        self.assertIn("BASE/'recovery_v1/audit.py'",transport.common_source())
    def test_prior_directory_including_partial_blocks(self):
        first=transport.rescue_jobs()[0][0]['job_id']
        original=Path.iterdir
        def altered(path):
            rows=list(original(path))
            if path==HERE.parent/'production/shard_0000/routes':rows.append(path/first)
            return iter(rows)
        with mock.patch.object(Path,'iterdir',altered):
            with self.assertRaisesRegex(AssertionError,'ORIGINAL_ROUTE_ATTEMPTED'):transport.rescue_jobs()
    def test_prior_ledger_terminal_blocks(self):
        import json
        first=transport.rescue_jobs()[0][0]['job_id']
        original=Path.read_bytes
        def altered(path):
            raw=original(path)
            if path==HERE.parent/'production/shard_0000/LEDGER.jsonl':raw+=(json.dumps(dict(job_id=first,status='REJECTED'))+'\n').encode()
            return raw
        with mock.patch.object(Path,'read_bytes',altered):
            with self.assertRaisesRegex(AssertionError,'ORIGINAL_ROUTE_ATTEMPTED'):transport.rescue_jobs()
    def test_finite_holder_environment_restoration_kept(self):
        source=transport.run_source()
        self.assertIn("identity.get('project_cache_environment',{})",source)
        self.assertIn('holder_environment_matches(identity,pid)',source)
        self.assertIn('holder_command(identity)',source)
    def test_merge_scope_explicit(self):
        self.assertIn('UNATTEMPTED_RESCUE_SUBSET_ONLY',transport.merge_source())


if __name__=='__main__':unittest.main(verbosity=2)
