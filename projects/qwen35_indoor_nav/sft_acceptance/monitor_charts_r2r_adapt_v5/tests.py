import importlib.util,json,subprocess,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('r2r_monitor_tests',HERE/'server.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_api(self):
        d=m.collect(False);self.assertEqual(d['monitor_version'],'ordinary_r2r_adapt_v5')
        self.assertEqual(d['current_segment_planned_decisions'],396001)
        self.assertEqual(d['navigation']['matched']['result']['sr'],.21)
        self.assertEqual(d['navigation']['full_epoch']['result']['sr'],.14)
        self.assertFalse(d['controller_globally_enabled']);json.dumps(d,allow_nan=False)
    def test_html(self):
        t=m.html().decode();self.assertIn('navigation-results',t);self.assertIn('R2R',t)
        self.assertNotIn('onclick=',t)
    def test_js(self):
        r=subprocess.run(['node','--check',str(HERE/'refresh.js')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_null_eta(self):
        self.assertIn('Number.isFinite(d.estimated_segment_remaining_seconds)',(HERE/'refresh.js').read_text())
    def test_live_plot_field(self):
        d=m.collect(False);self.assertTrue(d['points'])
        self.assertIn('rolling_ce',d['points'][-1])
        self.assertIn('x.rolling_ce',(HERE/'refresh.js').read_text())
if __name__=='__main__':unittest.main()
