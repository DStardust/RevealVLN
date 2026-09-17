import importlib.util
from pathlib import Path
import re
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('continue_monitor_test',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_correct_pool(self):
        d=m.collect(False);self.assertEqual(d['snapshot_counts']['instruction_conditioned_decisions'],2650347)
        self.assertEqual(d['epoch_total_decisions'],2650347)

    def test_incremental_and_cumulative_separate(self):
        d=m.collect(False)
        self.assertEqual(d['training_completion']['cumulative_expanded_updates']-4000,d['current_segment_new_updates'])
        self.assertEqual(d['decisions']-340712,d['current_segment_new_decisions'])

    def test_not_automatically_continued_again(self):
        d=m.collect(False);self.assertFalse(d['automatic_training']);self.assertFalse(d['controller_globally_enabled']);self.assertEqual(d['navigation']['high_lr']['result']['sr'],.14)

    def test_old_matched_metrics_visible(self):
        d=m.collect(False);self.assertEqual(d['navigation']['matched']['result']['sr'],.21)
        self.assertEqual(d['navigation']['before']['result']['sr'],.14)

    def test_old_51301_curve_not_joined(self):
        points=m.collect(False)['points']
        self.assertTrue(any(p['segment']=='expanded_stage1' for p in points))
        self.assertTrue(all(p['segment']!='legacy_v3' for p in points))
        first={}
        for p in points:first.setdefault(p['segment'],p)
        self.assertTrue(all(p['discontinuity'] for p in first.values()))

    def test_ui_single_fetch(self):
        h=m.html().decode();self.assertEqual(h.count('fetch('),1)
        self.assertNotIn('failure-results',h);self.assertIn('本段新增',h)

    def test_script_syntax(self):
        h=m.html().decode();scripts='\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S))
        result=subprocess.run(['node','--check','-'],input=scripts,text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
