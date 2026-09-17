import ast
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('calibration_monitor_test',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def test_source_syntax(self):ast.parse((HERE/'server.py').read_text())
    def test_historical_failures_preserved(self):
        d=m.collect(False)
        self.assertEqual(d['navigation']['matched']['result']['sr'],.21)
        self.assertEqual(d['navigation']['high_lr']['result']['sr'],.14)
        self.assertEqual(d['navigation']['continued']['result']['sr'],.13)
    def test_no_training_restart(self):
        d=m.collect(False)
        self.assertTrue(d['training_completion']['completed_budget'])
        self.assertTrue(d['training_completion']['holder_restoration_verified'])
    def test_json(self):json.dumps(m.collect(False),allow_nan=False)
    def test_html_and_js(self):
        text=m.html().decode();self.assertIn('stop-calibration-progress',text)
        self.assertIn('navigation-results',text)
        node=shutil.which('node');self.assertTrue(node)
        for script in re.findall(r'<script[^>]*>(.*?)</script>',text,re.S):
            r=subprocess.run([node,'--check','-'],input=script,text=True,capture_output=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
