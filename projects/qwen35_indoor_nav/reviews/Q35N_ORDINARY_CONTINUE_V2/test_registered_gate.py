"""Test actual frozen gate expression at boundaries without running navigation."""
import ast
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'workflow.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='close_report')
assignment=next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='positive' for t in n.targets))
expression=compile(ast.Expression(assignment.value),'frozen_gate','eval')
def gate(sr,spl,ndtw):return eval(expression,{'__builtins__':{}},{'delta':dict(success=sr,spl=spl,ndtw=ndtw)})


class GateTests(unittest.TestCase):
    def test_all_positive(self):self.assertTrue(gate(.01,.01,.01))
    def test_unchanged_sr_not_positive(self):self.assertFalse(gate(0,.02,.02))
    def test_sr_regression_not_positive(self):self.assertFalse(gate(-.01,.02,.02))
    def test_spl_drop_not_positive(self):self.assertFalse(gate(.03,-.000001,.01))
    def test_registered_ndtw_boundary(self):self.assertTrue(gate(.01,0,-.01))
    def test_beyond_ndtw_boundary(self):self.assertFalse(gate(.03,.01,-.010001))
    def test_perfect_stall_reduction_cannot_substitute_success(self):self.assertFalse(gate(0,0,0))


if __name__=='__main__':unittest.main()
