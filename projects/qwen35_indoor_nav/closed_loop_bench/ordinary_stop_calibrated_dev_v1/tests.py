import ast
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('stop_dev_source_test',HERE/'reuse.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


class IntegrationTests(unittest.TestCase):
    def test_all_sources_compile(self):
        for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):
            ast.parse(r.source(name))
    def test_simulator_metrics_transport_unchanged(self):
        for name in ('common.py','executor.py','path_metrics.py','official_distance.py','launch.py'):
            self.assertEqual(r.source(name),r.parent.source(name))
    def test_policy_change_has_only_logit_inputs(self):
        tree=ast.parse(r.source('evaluate.py'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='choose']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0]),"_stop.choose(raw_values, p['stop_logit_bias'])")
    def test_raw_and_calibrated_logits_logged(self):
        text=r.source('evaluate.py')
        self.assertIn("raw_logits=raw_values,stop_logit_bias=p['stop_logit_bias']",text)
        self.assertIn('batch_size=1  # Matched comparison',text)
    def test_transport_race_fix_retained(self):
        text=r.source('launch.py')
        self.assertIn('size,transient_misses=_scan.tree_size(OUT)',text)
        self.assertIn('GPU_NOT_EMPTY',text)


if __name__=='__main__':unittest.main()
