import unittest
import balanced_v2 as v2


class V2Tests(unittest.TestCase):
    def test_canonical_control_type(self):
        self.assertEqual(v2.CONTROL_TYPE,'completed_subgoal_revisit_placement_not_event_free_detour')

    def test_moving_i_count_balance(self):
        rows=v2.matched(list('FLR'),list('FRL'),list('FLFR'),2,4)
        self.assertEqual(len({tuple(r.count(a) for a in 'FLR') for r in rows.values()}),1)
        self.assertEqual(len({tuple(r) for r in rows.values()}),3)

    def test_turn_only_i_denied(self):
        with self.assertRaises(v2.Reject):v2.matched(list('FLR'),list('FRL'),list('LR'),2,2)

    def test_enumeration_still_bounded(self):
        accepted,ledger=v2.solve(list('FLR'),list('FRL'),list('FLFR'))
        self.assertEqual(len(ledger),240)
        self.assertGreater(len(accepted),0)

    def test_original_checker_unchanged(self):
        self.assertIs(v2.BalancedFactory.validate_matrix,v2.FamilyFactory.validate_matrix)
        self.assertIs(v2.BalancedFactory.replay_seeds,v2.FamilyFactory.replay_seeds)


if __name__=='__main__':unittest.main()
