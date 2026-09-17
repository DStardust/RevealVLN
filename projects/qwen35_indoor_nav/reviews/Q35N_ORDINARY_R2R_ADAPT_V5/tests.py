import ast,importlib.util,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('r2r_workflow_tests',HERE/'workflow.py')
w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Tests(unittest.TestCase):
    def test_gate(self):
        for delta,ok in [({'success':.01,'spl':0,'ndtw':-.01},True),({'success':0,'spl':1,'ndtw':1},False),({'success':.01,'spl':-.00001,'ndtw':0},False),({'success':.01,'spl':0,'ndtw':-.010001},False)]:
            self.assertEqual(w.positive_gate(delta),ok)
    def test_paths(self):
        self.assertEqual(w.TRAIN.name,'ordinary_r2r_adapt_v5')
        self.assertEqual(w.CASE.name,'ordinary_r2r_adapt_dev_v5')
        self.assertEqual(w.HERE,HERE)
    def test_python_sources(self):
        for p in HERE.glob('*.py'):ast.parse(p.read_text())
    def test_checkpoint_and_no_retry(self):
        self.assertIn('accept_final.py',w.text)
        self.assertIn('checkpoint_000004200',w.text)
        self.assertIn("final['cursor']['updates']==8000",w.text)
if __name__=='__main__':unittest.main()
