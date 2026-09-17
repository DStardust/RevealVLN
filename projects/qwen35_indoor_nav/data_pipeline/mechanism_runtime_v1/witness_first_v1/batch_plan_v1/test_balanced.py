import collections
import copy
import unittest
from balanced import matched,solve,BalancedFactory,FamilyFactory,Reject,closed


class BalancedTests(unittest.TestCase):
    def fixture(self):return list('FLR'),list('FRL'),list('LR')

    def test_equal_counts(self):
        rows=matched(*self.fixture(),2,2)
        self.assertEqual(len({tuple(actions.count(t) for t in 'FLR') for actions in rows.values()}),1)
        self.assertEqual(len({tuple(actions) for actions in rows.values()}),3)

    def test_exact_placement(self):
        a,b,i=self.fixture();rows=matched(a,b,i,2,2)
        self.assertEqual(rows['H_A'],a+i+a)
        self.assertEqual(rows['H_A_I'],a+a+i)
        self.assertEqual(rows['H_B'],b+b+['L','R'])

    def test_no_source_changes(self):
        source=self.fixture();saved=copy.deepcopy(source);matched(*source,2,2)
        self.assertEqual(source,saved)

    def test_forward_count_reject(self):
        with self.assertRaises(Reject):matched(*self.fixture(),2,3)

    def test_net_turn_reject(self):
        with self.assertRaises(Reject):matched(list('FL'),list('FR'),list('LR'),2,2)

    def test_duplicates_reject(self):
        with self.assertRaises(Reject):matched(list('F'),list('F'),list('FF'),2,4)

    def test_history_limit(self):
        with self.assertRaises(Reject):matched(list('F'*300+'LR'),list('F'*300+'RL'),list('LR'),2,2)

    def test_bad_actions(self):
        with self.assertRaises(Reject):matched(['S'],['F'],['L'],2,2)

    def test_multiplier_bounds(self):
        for k,j in [(1,2),(17,1),(2,0),(2,17),(True,1)]:
            with self.assertRaises(Reject):matched(*self.fixture(),k,j)

    def test_full_enumeration_and_determinism(self):
        plans,ledger=solve(*self.fixture())
        self.assertEqual(len(ledger),240)
        self.assertEqual((plans,ledger),solve(*self.fixture()))
        self.assertEqual((plans[0]['k'],plans[0]['j']),(2,2))

    def test_original_matrix_unmodified(self):
        self.assertIs(BalancedFactory.validate_matrix,FamilyFactory.validate_matrix)
        self.assertIs(BalancedFactory.replay_seeds,FamilyFactory.replay_seeds)
        self.assertIs(BalancedFactory.construct,FamilyFactory.construct)

    def test_continuations_verbatim(self):
        obj=object.__new__(BalancedFactory)
        obj.components={'a':list('FLR'),'b':list('FRL'),'i':list('LR'),'terminal':list('FL')}
        out=obj.continuations(None,None)
        self.assertEqual(out,{'C0':list('FLS'),'C_A':list('FLRFLS'),'C_B':list('FRLFLS')})

    def test_closure_original_tolerance(self):
        p={'position':[0,0,0],'rotation':[1,0,0,0]}
        first={**p,'sensors':{'rgb':copy.deepcopy(p),'semantic':copy.deepcopy(p)}}
        self.assertTrue(closed(first,first))
        last=copy.deepcopy(first);last['position'][0]=1.1e-5
        self.assertFalse(closed(first,last))


if __name__=='__main__':unittest.main()
