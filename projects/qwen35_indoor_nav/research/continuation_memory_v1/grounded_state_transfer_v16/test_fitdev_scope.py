"""Focused regressions for the authorized FIT/DEV scope."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

from build_data import FITDEV_SCOPE,FORMAL_SCOPE,scope_admission
from evaluate_continuations import registry_value
import objective
from train import schedule_for_seed,training_families
from v16_common import c


def families(split,count):
    return [dict(family_id=f'{split}_{i:02d}',house=f'{split}_house',split=split) for i in range(count)]


class FitDevScopeTest(unittest.TestCase):
    def setUp(self):
        self.rows=families('FIT',16)+families('DEV',2)
        self.config=dict(scope=FITDEV_SCOPE,authorized_training_families=sorted(r['family_id'] for r in self.rows if r['split']=='FIT'),
            authorized_diagnostic_families=sorted(r['family_id'] for r in self.rows if r['split']=='DEV'),
            seeds=[1209,1210,1211],arms=['B1','B2','Ours'])

    def test_fit16_dev2_admitted_but_incomplete_or_test_rejected(self):
        admitted=scope_admission(self.rows,FITDEV_SCOPE)
        self.assertEqual(admitted['training_admission'],'FIT16_DEV2_TRAINING_AUTHORIZED')
        data=dict(families=self.rows,training_admission=admitted['training_admission'])
        self.assertEqual(len(training_families(data,self.config)),16)
        with self.assertRaisesRegex(ValueError,'FITDEV_FAMILY_COUNT_OR_SPLIT'):
            scope_admission(self.rows[:-1],FITDEV_SCOPE)
        with self.assertRaisesRegex(ValueError,'FITDEV_FAMILY_COUNT_OR_SPLIT'):
            scope_admission(self.rows+families('TEST',1),FITDEV_SCOPE)
        formal=scope_admission(self.rows+families('TEST',8),FORMAL_SCOPE)
        self.assertTrue(formal['official_test_scale_complete'])

    def test_training_schedule_and_initial_state_are_fixed(self):
        ids=self.config['authorized_training_families'];ordinary=[[0,1],[2,3],[4]]
        left=schedule_for_seed(ids,ordinary,set(range(5)),1209);right=schedule_for_seed(ids,ordinary,set(range(5)),1209)
        self.assertEqual(left,right);self.assertEqual(len(left),1200);self.assertEqual(set(r['family'] for r in left),set(ids))
        self.assertTrue(all(i in set(range(5)) for row in left for i in row['ordinary']))
        self.assertEqual(c.model_identity(objective.initialize(1209))['sha256'],c.model_identity(objective.initialize(1209))['sha256'])

    def test_dev_20_180_and_formal_test_80_720(self):
        dev=registry_value(families('DEV',2),dict(self.config,evaluation_scope='DEV_DIAGNOSTIC'))
        self.assertEqual((len(dev['conditions']),len(dev['slots'])),(20,180))
        self.assertEqual((dev['main_rollout_slots'],dev['control_rollout_slots']),(144,36))
        self.assertEqual((dev['main_denominator_per_arm'],dev['control_denominator_per_arm']),(48,12))
        formal=registry_value(families('TEST',8),dict(self.config,evaluation_scope='FORMAL_TEST'))
        self.assertEqual((len(formal['conditions']),len(formal['slots'])),(80,720))


if __name__=='__main__':unittest.main(verbosity=2)
