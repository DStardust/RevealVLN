"""Bounded CPU tests; no GPU commands, renderers, or worker execution."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=load('bulk_prepare_cpu',HERE/'prepare.py')
a=load('bulk_adapters_cpu',HERE/'adapters.py')
r=load('bulk_runtime_cpu',HERE/'runtime.py')

class Tests(unittest.TestCase):
    def test_actual_manifest_order_remaining37(self):
        closed=[p.read(x) for x in p.CLOSED]
        excluded={x['house_id'] for d in closed for x in d['houses']}
        rows=p.choose(p.read(p.MANIFEST),set(p.read(p.SPLIT)['FIT']),excluded)
        self.assertEqual(len(rows),37)
        old=p.read(p.WF/'scout_next_v1/shard_1/PREPARED_CONFIG.json')['candidates']
        self.assertEqual([x['house_id'] for x in rows[:3]],[x['house_id'] for x in old])
    def test_transformed_sources_compile(self):
        for fn in (a.common_source,a.worker_source,a.recipe_source,a.supervisor_source):compile(fn(0),'<CPU_DRAFT>','exec')
    def test_shared_store_scope_does_not_cross_shards(self):
        for i in (0,1):
            common=r.module('cpu_store_'+str(i),a.common_source(i),a.SCOUT/'common.py')
            cls=common.store_class()
            self.assertEqual(cls.__init__.__globals__['FEEDBACK_ROOT'],a.shard_root(i))
    def test_original_quality_and_movement_logic_unchanged(self):
        source=a.worker_source(0)
        for marker in ("max_actions=140","if len(actions)>504:","bank.propose(records,max_programs=32)"):
            self.assertIn(marker,source)
        self.assertIn("module.FEEDBACK_ROOT=HERE",a.common_source(0))
    def test_exact_substitution_rejects_zero_or_duplicate(self):
        with self.assertRaises(ValueError):a.exact('a','b','c')
        with self.assertRaises(ValueError):a.exact('aa','a','b')
    def test_only_approved_first_gpu1_shard_available(self):
        with self.assertRaises(ValueError):a.supervisor_source(1)
        with self.assertRaises(ValueError):a.worker_source(0,5)
        with self.assertRaises(ValueError):a.shard_root(6)
    def test_recipe_uses_completed_bank_and_unchanged_caps(self):
        source=a.recipe_source(0)
        self.assertIn(str(a.shard_root(0)/'run_v1'),source)
        for marker in ('SCOUT_COMPONENT_BANK_COMPLETE','PARTIAL_BANK_LINE','max_programs=1000','if max_cont>160:'):
            self.assertIn(marker,source)
    def test_original_guard_byte_identical(self):
        old=a.checked(a.RUNTIME/'compact_loop_v2/run.py');guard=old[old.index('def check_gpu('):old.index('def disk_size(')]
        self.assertIn(guard,a.supervisor_source(0));self.assertIn("sample['elapsed'] < 4500",a.supervisor_source(0))
    def test_unapproved_common_fails_closed(self):
        common=r.module('cpu_unapproved',a.common_source(0),a.SCOUT/'common.py')
        with self.assertRaises(RuntimeError):common.runtime_config()

if __name__=='__main__':unittest.main()
