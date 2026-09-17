import importlib.util
from pathlib import Path
import collections
import unittest
spec=importlib.util.spec_from_file_location('revisit_test_method',Path(__file__).with_name('method.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_real_program_component_counts(self):
        import json
        cfg=json.loads((Path(__file__).parent.parent/'assembly_v1/run_v1/EXECUTION_CONFIG.json').read_text())
        c=cfg['candidates'][0]['components'];rows,i=m.matched_revisit(c['a'],c['b'],c['i'])
        self.assertEqual(len({tuple(sorted(collections.Counter(v).items())) for v in rows.values()}),1)
        self.assertEqual(len({tuple(v) for v in rows.values()}),3)
        self.assertGreaterEqual(i.count('F'),2)
        self.assertLessEqual(len(rows['H_A'])+8,512)
    def test_imbalanced_forward_rejected(self):
        with self.assertRaisesRegex(m.Reject,'FORWARD_COUNTS'):
            m.matched_revisit(list('FFLR'),list('FFFFLR'),list('LR'))
    def test_validator_not_overridden(self):
        self.assertIs(m.RevisitFactory.validate_matrix,m.WitnessFactory.validate_matrix)
        self.assertIs(m.RevisitFactory.replay_seeds,m.WitnessFactory.replay_seeds)
if __name__=='__main__':unittest.main()
