"""Exercise the actual final-report gate expression, including boundary cases."""
import ast
from pathlib import Path
import unittest


class GateTests(unittest.TestCase):
    def test_registered_expression_boundaries(self):
        tree=ast.parse(Path(__file__).with_name('audit_and_report.py').read_text())
        matches=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='gate' for t in n.targets)]
        self.assertEqual(len(matches),1)
        code=compile(ast.Expression(matches[0]),'registered-gate','eval')
        cases=[((.01,0,0),True),((0,.01,.01),False),((-.01,.01,.01),False),
               ((.01,-.000001,.01),False),((.01,0,-.01),True),
               ((.01,0,-.010001),False),((.03,.02,.01),True)]
        for values,expected in cases:
            with self.subTest(values=values):
                self.assertEqual(eval(code,{'__builtins__':{}},{'delta':dict(zip(('success','spl','ndtw'),values))}),expected)


if __name__=='__main__':unittest.main()
