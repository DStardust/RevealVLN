"""Validate actual compiler certificates, including unknown evidence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibration_r2 import event_label,summarize,legacy
from shared import read,LINE,HERE

class EventCalibration(unittest.TestCase):
    def test_unknown_empty_and_witnesses(self):
        self.assertIsNone(event_label(None))
        self.assertEqual(event_label([]),0)
        self.assertEqual(event_label([0]),1)
        self.assertEqual(event_label([42,91]),1)
        with self.assertRaises(ValueError):event_label('unknown')
        self.assertEqual(summarize([(0.,0),(1.,1)])['brier'],0)
        self.assertEqual(summarize([(0.,0),(1.,1)])['ece'],0)

    def test_real_training_certificate_agreement(self):
        data=read(HERE/'runs/gpu_001/DATA.json');family=data['raw_families'][0]
        compiler=legacy.Compiler(**family['compiler'])
        trace=read(LINE/next(iter(family['traces'].values()))['path'])
        atoms=compiler.atoms(trace['observations'])
        labels=[event_label(v) for row in atoms for v in row.values()]
        self.assertIn(1,labels);self.assertIn(0,labels)
        for row in atoms:
            for value in row.values():
                if value is not None:self.assertEqual(event_label(value),int(bool(value)))

if __name__=='__main__':unittest.main()
