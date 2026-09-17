import ast
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    def test_exact_allowed_source_and_scope_changes_only(self):
        old=(HERE.parent/'prepare.py').read_text()
        expected=old.replace('WF=HERE.parent','WF=HERE.parents[1]').replace('sys.path.insert(0,str(HERE))','sys.path.insert(0,str(HERE.parent))')
        expected=expected.replace("source=WF/'scout_v1/run_v1'","source=WF/'scout_next_v1/shard_0/run_v1'")
        expected=expected.replace("HERE/'method.py',HERE/'prepare.py'","HERE.parent/'method.py',HERE/'prepare.py'")
        expected=expected.replace("if output.exists() or not output.is_relative_to(HERE) or output==HERE:raise ValueError('FRESH_SNAPSHOT_REQUIRED')", "if output.exists() or output.parent!=HERE or not output.name.startswith('snapshot'):raise ValueError('FRESH_SNAPSHOT_REQUIRED')")
        self.assertEqual(expected.strip(),(HERE/'prepare.py').read_text().strip())

    def test_syntax(self):ast.parse((HERE/'prepare.py').read_text())

    def test_original_winding_method_is_referenced(self):
        src=(HERE/'prepare.py').read_text()
        self.assertIn('from method import solve,matched,closed,FamilyFactory,CONTROL_TYPE',src)
        self.assertIn("HERE.parent/'method.py'",src)

    def test_no_permit_or_gpu_in_source_adapter(self):
        src=(HERE/'prepare.py').read_text()
        self.assertIn("'runtime_allowed':False",src)
        self.assertIn("'gpu_device':None",src)
        self.assertNotIn('HabitatBackend(',src)

    def test_complete_house_guard_preserved(self):
        src=(HERE/'prepare.py').read_text()
        self.assertIn("terminal['status']!='SCOUT_COMPONENT_BANK_COMPLETE'",src)
        self.assertIn("'HOUSE_STILL_RUNNING_NO_BANK_READ'",src)


if __name__=='__main__':unittest.main()
