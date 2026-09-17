import copy
import unittest
import method as m


class WindingTests(unittest.TestCase):
    def fixture(self):return list('FF'+'L'*24),list('FFF'+'R'*24),list('F'+'L'*12+'F'+'L'*12)

    def test_winding_count_match(self):
        p=m.plan(*self.fixture(),2,2)
        self.assertEqual(p['required_neutral_spin_directions'],['L','R'])
        self.assertEqual(p['action_counts'],{'F':6,'L':72,'R':48})
        self.assertEqual(p['pads']['H_A']['revolutions'],2)
        self.assertEqual(p['pads']['H_B']['revolutions'],3)
        self.assertEqual(p['history_actions'],126)

    def test_original_action_prefixes(self):
        a,b,i=self.fixture();p=m.plan(a,b,i,2,2)
        self.assertEqual(p['histories']['H_A'][:len(a+i+a)],a+i+a)
        self.assertEqual(p['histories']['H_A_I'][:len(a+a+i)],a+a+i)
        self.assertEqual(p['histories']['H_B'][:len(b+b)],b+b)

    def test_noninteger_winding_rejected(self):
        a,b,i=self.fixture();a+=['L']
        with self.assertRaises(m.Reject):m.plan(a,b,i,2,2)

    def test_forward_mismatch(self):
        with self.assertRaises(m.Reject):m.plan(*self.fixture(),3,2)

    def test_length_still_bounded(self):
        a,b,i=self.fixture();a+=['L']*240
        with self.assertRaises(m.Reject):m.plan(a,b,i,2,2)

    def test_no_turn_only_i(self):
        with self.assertRaises(m.Reject):m.plan(list('FLR'),list('FRL'),list('LR'),2,2)

    def test_240_attempts(self):
        accepted,ledger=m.solve(*self.fixture())
        self.assertEqual(len(ledger),240)
        self.assertGreater(len(accepted),0)

    def test_independent_maxima_need_alignment(self):
        a=list('F'+'L'*30+'R'*6);b=list('FF'+'L'*25+'R'*25);i=list('FF')
        p=m.plan(a,b,i,2,2)
        self.assertEqual(p['target_alignment']['original_max_L'],60)
        self.assertEqual(p['target_alignment']['original_max_R'],50)
        self.assertEqual(p['target_alignment']['raise_direction'],'R')
        self.assertEqual(p['target_alignment']['raise_steps'],10)
        self.assertEqual(p['action_counts'],{'F':4,'L':60,'R':60})

    def test_alignment_tie_prefers_left(self):
        p=m.plan(list('F'+'L'*18+'R'*6),list('FF'+'L'*12+'R'*12),list('FF'),2,2)
        self.assertEqual(p['target_alignment']['raise_steps'],12)
        self.assertEqual(p['target_alignment']['raise_direction'],'L')

    def test_fast_filter_preserves_brute_ledger(self):
        a,b,i=self.fixture();_,ledger=m.solve(a,b,i)
        brute=[]
        for k in range(2,17):
            for j in range(1,17):
                try:m.plan(a,b,i,k,j);reason='COUNT_PLAN_REQUIRES_REAL_SPIN_VALIDATION'
                except m.Reject as error:reason=error.code
                brute.append([k,j,reason])
        self.assertEqual(ledger,brute)

    def test_not_mutated(self):
        x=self.fixture();saved=copy.deepcopy(x);m.solve(*x);self.assertEqual(x,saved)

    def test_original_checker_and_alias(self):
        self.assertIs(m.BalancedFactory,m.WindingBalancedFactory)
        self.assertIs(m.BalancedFactory.validate_matrix,m.FamilyFactory.validate_matrix)
        self.assertIs(m.BalancedFactory.replay_seeds,m.FamilyFactory.replay_seeds)

    def fake(self,forbidden=False,collision=False,closure=False):
        pose={'position':[0.,0.,0.],'rotation':[1.,0.,0.,0.]}
        initial={**copy.deepcopy(pose),'sensors':{'rgb':copy.deepcopy(pose),'semantic':copy.deepcopy(pose)}}
        trace={'trace_hash':'a'*64,'complete':True,'collisions':int(collision),'observations':[{'pose':initial}]}
        if closure:
            trace=copy.deepcopy(trace);trace['observations'][0]['pose']['position'][0]=.01
        obj=object.__new__(m.BalancedFactory);obj.a='anchor_A';obj.b='anchor_B'
        obj.padding_plan={'required_neutral_spin_directions':['L','R']};calls=[];events=[]
        class Runner:
            def run(self,position,yaw,actions):calls.append(actions);return trace
        obj.runner=Runner();obj.pattern=lambda tr,required,forbid: not forbidden and not tr['collisions']
        obj.emit=lambda kind,value:events.append((kind,value))
        return obj,initial,calls,events

    def test_required_spin_each_direction_replayed(self):
        obj,pose,calls,events=self.fake();e=obj.neutral_spins([0,0,0],0,pose)
        self.assertEqual(calls,[['L']*24,['R']*24]);self.assertEqual(set(e),{'L','R'})
        self.assertEqual(len(events),2)

    def test_spin_witness_failure_rejected_before_more(self):
        obj,pose,calls,events=self.fake(forbidden=True)
        with self.assertRaises(m.Reject):obj.neutral_spins([0,0,0],0,pose)
        self.assertEqual(len(calls),1);self.assertEqual(events,[])

    def test_spin_collision_rejected(self):
        obj,pose,_,_=self.fake(collision=True)
        with self.assertRaises(m.Reject):obj.neutral_spins([0,0,0],0,pose)

    def test_spin_closure_failure_rejected(self):
        obj,pose,_,_=self.fake(closure=True)
        with self.assertRaises(m.Reject):obj.neutral_spins([0,0,0],0,pose)

    def test_no_required_direction_no_spin(self):
        obj,pose,calls,_=self.fake();obj.padding_plan['required_neutral_spin_directions']=[]
        self.assertEqual(obj.neutral_spins([0,0,0],0,pose),{});self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
