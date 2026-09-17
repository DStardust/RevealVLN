import importlib.util,json,subprocess,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('data_mon_tests',HERE/'server.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_api(self):
        d=m.collect(False);self.assertEqual(d['monitor_version'],'ordinary_onpolicy_v6r1')
        self.assertEqual(d['navigation']['continued']['result']['sr'],.15)
        self.assertFalse(d['onpolicy_recovery']['training_started']);json.dumps(d,allow_nan=False)
    def test_html(self):self.assertIn('recovery-status',m.html().decode())
    def test_js(self):
        r=subprocess.run(['node','--check',str(HERE/'refresh.js')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
