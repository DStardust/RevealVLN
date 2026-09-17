"""CPU/API-data and JavaScript syntax acceptance; does not claim browser screenshots."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('monitor_under_test',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def test_completed_budget(self):
        r=dict(status='STOPPED',stop=['BUDGET:max_updates'],cursor=dict(updates=4000))
        self.assertTrue(m.classify_training(r,dict(holders_restored=True))['completed_budget'])

    def test_failure_not_completed(self):
        for status,stop,updates in [('STOPPED',['BUDGET:wall'],100),('FAILED',[],4000),('STOPPED',['BUDGET:max_updates'],200)]:
            self.assertFalse(m.classify_training(dict(status=status,stop=stop,cursor=dict(updates=updates)),{})['completed_budget'])

    def test_no_inferred_resource_restoration(self):
        self.assertFalse(m.classify_training({}, {})['holder_restoration_verified'])

    def test_actual_data_and_metrics(self):
        d=m.collect(False)
        self.assertEqual(d['snapshot_counts']['instruction_conditioned_decisions'],2650347)
        self.assertEqual(d['training_completion']['new_updates'],4000)
        self.assertEqual(d['decisions'],340712)
        self.assertEqual(d['navigation']['before']['result']['selected_batch_size'],1)
        self.assertEqual(d['navigation']['after_batch8']['result']['selected_batch_size'],8)
        self.assertFalse(d['automatic_training'])
        self.assertTrue(all(p.get('segment')!='legacy_v3' for p in d['points']))

    def test_pending_does_not_fabricate_metrics(self):
        d=m.eval_state('NONEXISTENT_READ_ONLY_CASE')
        self.assertIsNone(d['result'])
        self.assertEqual(d['status'],'NOT_STARTED')

    def test_no_stale_unmeasured_claim_or_duplicate_fetch(self):
        h=m.html().decode()
        self.assertNotIn('新模型结果未测不能填零',h)
        self.assertNotIn('expandedStatus()',h)
        self.assertEqual(h.count('fetch('),1)
        self.assertIn('refreshBusy',h)
        self.assertIn('AbortController',h)
        self.assertNotIn('setInterval(refresh',h)

    def test_dom_ids_and_js_syntax(self):
        h=m.html().decode()
        ids=re.findall(r'\bid="([^"]+)"',h)
        self.assertEqual(len(ids),len(set(ids)))
        for name in ('navigation-results','failure-results','paired-summary','targeted-state'):
            self.assertIn(name,ids)
        scripts='\n'.join(re.findall(r'<script>(.*?)</script>',h,re.S))
        result=subprocess.run(['node','--check','-'],input=scripts,text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
