import ast
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('pair_reuse',HERE/'reuse_eval.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class Pair(unittest.TestCase):
    def test_hash_and_compile(self):
        for name in m.HASHES:compile(m.source(name),name,'exec')

    def test_policy_exact(self):self.assertEqual(m.source('evaluate.py'),(m.OLD/'evaluate.py').read_text())
    def test_simulator_exact(self):self.assertEqual(m.source('executor.py'),(m.OLD/'executor.py').read_text())
    def test_distance_exact(self):self.assertEqual(m.source('official_distance.py'),(m.OLD/'official_distance.py').read_text())
    def test_ndtw_exact(self):self.assertEqual(m.source('path_metrics.py'),(m.OLD/'path_metrics.py').read_text())

    def test_trace_audit_exact(self):
        marker='    overall=stats(rows);by_house='
        self.assertEqual(m.source('aggregate.py').split("    assert full, 'INCOMPLETE_DEV_CASE_NO_ADMITTED_METRICS'")[0],
                         (m.OLD/'aggregate.py').read_text().split(marker)[0])

    def test_only_training_presence_guard_removed(self):
        old="    assert all(x for x in training_before['processes']),'TRAINING_IDENTITY_NOT_PRESENT'"
        new="    # Training may not yet have started or already naturally stopped; GPU1 independent."
        self.assertEqual(m.source('launch.py').replace(new,old),(m.OLD/'launch.py').read_text())

    def test_selection(self):
        selection=json.loads((HERE/'SELECTION.json').read_text())
        es=json.loads((HERE/'EPISODES_PRIVILEGED.json').read_text())
        self.assertEqual(len(es),100);self.assertEqual(len(set(selection['physical_keys'])),100)
        self.assertEqual(len(set(e['episode_id'] for e in es)),100)
        self.assertFalse(selection['selection_depends_on_model_results'])
        split=json.loads((HERE.parents[1]/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json').read_text())
        self.assertEqual(set(selection['houses']),set(split['INTERNAL_DEV']))
        self.assertTrue(set(selection['houses']).isdisjoint(split['FIT']+split['INTERNAL_CONFIRM']))

    def test_before_and_after_same_source_wrappers(self):
        for name in m.HASHES:
            a=HERE.parent/'ordinary_expanded_dev_before_v1'/name;b=HERE.parent/'ordinary_expanded_dev_after_v1'/name
            self.assertEqual(a.read_bytes(),b.read_bytes())


if __name__=='__main__':unittest.main()
