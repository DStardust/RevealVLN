import ast
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('full_workflow_test',HERE/'workflow.py')
w=importlib.util.module_from_spec(s);s.loader.exec_module(w)


class WorkflowTests(unittest.TestCase):
    def test_compiled_source(self):ast.parse(w.text)
    def test_full_endpoint_and_bounds(self):
        for text in ('time.monotonic()-start<15000',"final['status']=='EPOCHS_COMPLETED'",'checkpoint_000008200.pt.json',"str(HERE/'accept_final.py')",'EVALUATING_FINAL_31059'):
            self.assertIn(text,w.text)
    def test_gate_boundaries(self):
        tree=ast.parse(w.CLOSE_REPORT)
        expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='positive' for t in n.targets))
        code=compile(ast.Expression(expr),'full-epoch-gate','eval')
        for vals,want in [((.01,0,0),True),((0,.01,0),False),((.01,-1e-9,0),False),((.01,0,-.01),True),((.01,0,-.010001),False)]:
            self.assertEqual(eval(code,{'__builtins__':{}},{'delta':dict(zip(('success','spl','ndtw'),vals))}),want)
    def test_own_identity_cleanup_only(self):
        self.assertIn("assert identity(owner['pid'])==owner",w.text)
        self.assertIn('foreign_processes_signaled=[]',w.text)


if __name__=='__main__':unittest.main()
