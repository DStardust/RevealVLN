import ast
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('full_epoch_monitor_test',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def test_source(self):ast.parse((HERE/'server.py').read_text())
    def test_full_data_and_no_controller(self):
        d=m.collect(False)
        self.assertEqual(d['source_checkpoint_updates'],8000)
        self.assertEqual(d['current_segment_planned_decisions'],1967696)
        self.assertFalse(d['controller_globally_enabled'])
        self.assertFalse(d['automatic_next_epoch'])
    def test_failures_not_hidden(self):
        d=m.collect(False)
        for k,want in [('matched',.21),('high_lr',.14),('continued',.13),('stop_calibrated',.16)]:
            self.assertEqual(d['navigation'][k]['result']['sr'],want)
    def test_json(self):json.dumps(m.collect(False),allow_nan=False)
    def test_html_js(self):
        text=m.html().decode();self.assertIn('navigation-results',text);self.assertIn('31,059',text)
        node=shutil.which('node');self.assertTrue(node)
        for script in re.findall(r'<script[^>]*>(.*?)</script>',text,re.S):
            r=subprocess.run([node,'--check','-'],input=script,text=True,capture_output=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
