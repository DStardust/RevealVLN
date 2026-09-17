import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_audit',HERE.parent/'audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class InterruptedStubTest(unittest.TestCase):
    def test_no_training_rows_from_interrupted_terminal(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder)
            d=root/'routes'/'fixture'
            d.mkdir(parents=True)
            with (d/'result.json').open('x') as handle:
                json.dump(dict(job_id='fixture',scene_id='test',status='RESOURCE_INTERRUPTED',training_eligible=False),handle)
            self.assertEqual(audit.audit_route(root,dict(job_id='fixture',scene_id='test')),([],set(),0))

if __name__=='__main__': unittest.main()
