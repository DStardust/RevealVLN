import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transport as t
import run


class TransportTests(unittest.TestCase):
    def test_exact_rejects_missing_or_duplicate(self):
        for source in ('none', 'oldold'):
            with self.assertRaises(AssertionError): t.exact(source, 'old', 'new')

    def test_invalid_shards(self):
        for shard in (-1, 2, True, '0'):
            with self.assertRaises(AssertionError): t.shard_root(shard)

    def test_all_transformed_sources_compile(self):
        for shard in (0, 1):
            for function in (t.common_source, t.worker_source, t.supervisor_source):
                compile(function(shard), 'cpu_transport_test', 'exec')

    def test_supervisor_gpu_and_scope(self):
        for shard in (0, 1):
            m = t.module_from_source('test_scoped_supervisor', t.supervisor_source(shard), HERE / 'fixture.py')
            self.assertEqual(m.HERE, t.shard_root(shard))
            self.assertEqual(m.OUT, t.shard_root(shard) / 'run_v1')
            self.assertEqual(m.ENV, t.ENV)
            self.assertEqual(m.UUID, t.GPUS[shard][1])
            self.assertIn(f"'--shard','{shard}'", t.supervisor_source(shard))
            self.assertIn(f"'nvidia-smi','-i','{t.GPUS[shard][0]}'", t.supervisor_source(shard))

    def test_worker_gpu_and_scope(self):
        for shard in (0, 1):
            s = t.worker_source(shard)
            self.assertIn(f"HabitatBackend(candidate['scene_glb'],{t.GPUS[shard][0]},candidate['roles']", s)
            self.assertIn('from scout_next_scoped_common import (', s)
            self.assertIn('all_hub_summaries', s)

    def test_common_scope_and_store(self):
        for shard in (0, 1):
            m = t.module_from_source('test_common_' + str(shard), t.common_source(shard), t.SCOUT / 'common.py')
            self.assertEqual(m.HERE, t.shard_root(shard))
            self.assertEqual(m.RUNTIME, t.RUNTIME)
            self.assertEqual(m.ROOT, t.ROOT)
            cls = m.store_class()
            with tempfile.TemporaryDirectory(dir=t.shard_root(shard)) as folder:
                with cls(Path(folder) / 'content', 1024**2) as store: pass
                self.assertTrue(store.final_audit['audit_pass'])

    def config_data(self, shard, approved=True):
        folder=t.shard_root(shard)
        cfg=json.loads((folder/'PREPARED_CONFIG.json').read_text())
        approval=dict(approved=approved,shard=shard,gpu=t.GPUS[shard][0],
            prepared_input_lock_sha256='mockhash',runtime_input_lock_sha256='mockhash')
        return cfg, approval

    def test_approved_transition_both_flags(self):
        for shard in (0,1):
            cfg, approval=self.config_data(shard)
            old=copy.deepcopy(cfg)
            def read(path): return json.dumps(approval if 'APPROVAL' in path.name else cfg)
            with mock.patch.object(t,'verify_lock'),mock.patch.object(t,'sha',return_value='mockhash'),mock.patch.object(Path,'read_text',read):
                value=t.runtime_config(shard)
            self.assertTrue(value['runtime_allowed'] and value['executable'] and value['runtime_adapter_ready'])
            self.assertFalse(value['training_allowed'])
            self.assertEqual(old,cfg)

    def test_no_approval_no_output_or_gpu(self):
        with mock.patch.object(t,'runtime_config',side_effect=AssertionError('approval')),mock.patch.object(Path,'mkdir') as mkdir,mock.patch.object(t,'module_from_source') as module:
            with self.assertRaises(AssertionError): run.execute(0)
            mkdir.assert_not_called();module.assert_not_called()

    def test_cross_shard_approval_rejected(self):
        cfg,approval=self.config_data(0)
        approval['shard']=1
        with mock.patch.object(t,'verify_lock'),mock.patch.object(t,'sha',return_value='mockhash'),mock.patch.object(Path,'read_text',lambda p:json.dumps(approval if 'APPROVAL' in p.name else cfg)):
            with self.assertRaises(AssertionError):t.runtime_config(0)

    def test_graphics_resource_upper_bound(self):
        m=t.module_from_source('test_gpu_limits',t.supervisor_source(1),HERE/'fixture.py')
        self.assertEqual(m.check_gpu(dict(memory_mib=1300,processes={1:{'mib':246,'type':'C'},2:{'mib':1000,'type':'G'}}),2),1054)
        with self.assertRaises(AssertionError):m.check_gpu(dict(memory_mib=3000,processes={1:{'mib':2000,'type':'G'}}),2)


if __name__ == '__main__':unittest.main()
