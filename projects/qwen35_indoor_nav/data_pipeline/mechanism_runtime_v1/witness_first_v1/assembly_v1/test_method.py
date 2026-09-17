import collections
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from method import balanced_histories,WitnessFactory,FamilyFactory,Reject

class Tests(unittest.TestCase):
    def test_exact_counts_and_distinct_order(self):
        a=list('FLRFLR');b=list('LFFR');n=['F']+['L']*12+['F']+['R']*12;i=['L','R']+n
        rows=balanced_histories(a,b,i,n)
        self.assertEqual(len({tuple(sorted(collections.Counter(v).items())) for v in rows.values()}),1)
        self.assertEqual(len({tuple(v) for v in rows.values()}),3)
        self.assertEqual(rows['H_A'][:len(n)],n)
    def test_no_fake_turn_rebalancing(self):
        with self.assertRaisesRegex(Reject,'TURN_IMBALANCE'):
            balanced_histories(list('FFL'),list('FFR'),list('LR'),list('FFLR'))
    def test_no_zero_forward_control(self):
        with self.assertRaisesRegex(Reject,'NO_TRANSLATION'):
            balanced_histories(list('FFLR'),list('LFFR'),list('LR'),list('LR'))
    def test_full_certification_unchanged(self):
        for k in ('construct','validate_matrix','replay_seeds'):
            self.assertIs(getattr(WitnessFactory,k),getattr(FamilyFactory,k))
    def test_length_not_relaxed(self):
        with self.assertRaisesRegex(Reject,'HISTORY_LENGTH'):
            balanced_histories(list('FFLR')*130,list('LFFR')*130,list('LRFF'),list('FFLR'))
if __name__=='__main__':unittest.main()
