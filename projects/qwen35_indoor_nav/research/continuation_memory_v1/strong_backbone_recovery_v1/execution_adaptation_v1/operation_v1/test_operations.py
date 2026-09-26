"""CPU checks for denominator integrity, model admission and owned cleanup."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline
import review


class OperationsTests(unittest.TestCase):
    def test_partial_groups_never_produce_admission_sr(self):
        names = ['NATIVE', 'CURRENT_s42', 'DELTA_s42']
        entries = [dict(id=0, house='a'), dict(id=1, house='b')]
        rows = {0:dict(house='a',outcomes={a:dict(success=float(a != 'NATIVE'),spl=.5,steps=10) for a in names})}
        s = review.summarize(rows, entries, names, [42])
        self.assertFalse(s['evaluation_complete'])
        self.assertIsNone(s['arms']['DELTA_s42']['sr'])
        self.assertEqual(s['arms']['DELTA_s42']['sr_identification_bounds'], [.5, 1.])
        self.assertEqual(s['paired']['DELTA_s42_vs_NATIVE']['difference_identification_bounds'], [0., 1.])
        del rows[0]['outcomes']['CURRENT_s42']
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_MODEL_GROUP'):
            review.summarize(rows, entries, names, [42])

    def test_pair_success_and_regression_full_denominator(self):
        names = ['NATIVE','CURRENT_s42','DELTA_s42']
        entries = [dict(id=i,house='a') for i in range(2)]
        rows = {i:dict(house='a',outcomes={a:dict(success=float(a == 'DELTA_s42') if i == 0 else float(a != 'DELTA_s42'),spl=.4,steps=20) for a in names}) for i in range(2)}
        s = review.summarize(rows, entries, names, [42])
        self.assertTrue(s['evaluation_complete'])
        p = s['paired']['DELTA_s42_vs_CURRENT_s42']
        self.assertEqual(p['wins'], [0]); self.assertEqual(p['losses'], [1]); self.assertEqual(p['delta_sr'], 0.)

    def test_cleanup_refuses_wrong_owner_identity(self):
        class Proc:
            pid = os.getpid()
            def poll(self): return None
        w = dict(proc=Proc(), ticks='not-the-recorded-process')
        with patch.object(pipeline.os, 'getpgid', return_value=os.getpid()):
            self.assertFalse(pipeline.owned(w))
            w['ticks'] = pipeline.process_start(os.getpid())
            self.assertTrue(pipeline.owned(w))
        with patch.object(pipeline.os, 'getpgid', return_value=os.getpid()+1):
            self.assertFalse(pipeline.owned(w))

    def test_live_attempt_is_not_recovered_or_erased(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp); (root/'attempts').mkdir()
            path = root/'attempts/test.start.json'
            path.write_text(json.dumps(dict(pid=os.getpid(),ticks=pipeline.process_start(os.getpid()),started_unix=0,gpu=0,cost_phase='TRAIN')))
            with self.assertRaisesRegex(RuntimeError,'PREVIOUS_WORKER_STILL_RUNNING'):
                pipeline.unfinished_attempts(root)
            self.assertTrue(path.exists()); self.assertFalse((root/'attempts/test.json').exists())

    def test_final_weight_identity_required(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp); folder=root/'CURRENT_s42';folder.mkdir()
            (folder/'FINAL.pt').write_bytes(b'fixture-only-not-a-model')
            path=folder/'RESULT.json'
            value=dict(status='COMPLETE',completed_steps=3000,final_sha256=pipeline.u.sha(folder/'FINAL.pt'))
            path.write_text(json.dumps(value))
            p=dict(models={'CURRENT_s42':{}},training_steps=3000)
            self.assertEqual(pipeline.completed_models(root,p),['CURRENT_s42'])
            (folder/'FINAL.pt').write_bytes(b'changed-fixture')
            with self.assertRaisesRegex(ValueError,'TRAINED_MODEL_CHANGED'):
                pipeline.completed_models(root,p)


if __name__ == '__main__':
    unittest.main()
