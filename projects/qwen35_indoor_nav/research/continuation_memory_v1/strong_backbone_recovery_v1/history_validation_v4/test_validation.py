import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent),str(HERE.parent/'recovery_confirmation_v2')]
import torch
from mechanism import interventions, match_norm, unroll
from confirmation_model import ConfirmationMemory, sequence_logits
from confirmation_review import summarize
from prepare_unseen import pick
from transfer_audit import audit_pair


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1209)
        torch.set_num_threads(2)

    def head(self, architecture='CONCAT'):
        model = ConfirmationMemory(16,architecture)
        with torch.no_grad():
            model.actor[-1].weight.normal_(0,.2)
        return model.eval()

    def test_history_interventions_preserve_norm_and_recent_input(self):
        head = self.head()
        prefix = torch.randn(20,16)
        full = unroll(head,prefix)
        d = interventions(head,prefix,full)
        self.assertTrue(torch.equal(d['FULL'],d['SHAM']))
        self.assertFalse(torch.equal(d['RECENT8_RAW'],full))
        for k in ('RECENT8_NORM','REVERSED_OLD_NORM','CURRENT_NORM'):
            self.assertTrue(torch.allclose(d[k].norm(),full.norm(),rtol=1e-6,atol=1e-7))
        # The reverse control only reorders older causal observations.
        reverse = unroll(head,torch.cat((prefix[:-8].flip(0),prefix[-8:])))
        self.assertTrue(torch.equal(d['REVERSED_OLD_NORM'],match_norm(reverse,full)))
        self.assertFalse(torch.allclose(d['REVERSED_OLD_NORM'],full))

    def test_no_future_or_parameter_update_and_current_control(self):
        for architecture in ('CONCAT','LOCAL'):
            head = self.head(architecture)
            before = {k:v.clone() for k,v in head.state_dict().items()}
            features = torch.randn(24,16)
            with torch.no_grad():
                full = unroll(head,features[:16])
                a = interventions(head,features[:16],full)
                features[16:] = 1000
                b = interventions(head,features[:16],unroll(head,features[:16]))
                for k in a:
                    self.assertTrue(torch.equal(a[k],b[k]))
                    if architecture=='LOCAL':
                        self.assertTrue(torch.equal(a[k],full))
            self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in before.items()))

    def test_full_readout_matches_original_training_path(self):
        head = self.head()
        row = dict(memory_features=torch.randn(20,16),query_steps=torch.tensor([0,7,19]),
                   actor_features=torch.randn(3,16),base_logits=torch.randn(3,4))
        old = sequence_logits(head,row)
        actual = torch.cat([row['base_logits'][q:q+1]+head.action_delta(row['actor_features'][q:q+1],
            unroll(head,row['memory_features'][:t+1])) for q,t in enumerate(row['query_steps'])])
        self.assertTrue(torch.equal(old,actual))
        short = row['memory_features'][:8]
        d = interventions(head,short,unroll(head,short))
        self.assertTrue(torch.equal(d['FULL'],d['RECENT8_NORM']))

    def test_invalid_norm_not_silently_filled(self):
        with self.assertRaisesRegex(ValueError,'ZERO_DIRECTION'):
            match_norm(torch.zeros(4),torch.ones(4))
        with self.assertRaisesRegex(ValueError,'NONFINITE'):
            match_norm(torch.tensor([float('nan')]),torch.ones(1))

    def test_route_family_isolation_includes_other_instructions(self):
        episodes = [dict(scene_id=f'mp3d/{h}/{h}.glb',trajectory_id=r,episode_id=10*r+i)
                    for h in ('a','b') for r in range(4) for i in range(3)]
        chosen,audit = pick(episodes,3,1209,{'b'},{('a','0')})
        self.assertEqual({str(e['trajectory_id']) for e in chosen},{'1','2','3'})
        self.assertEqual(len(chosen),audit['eligible_distinct_routes'])
        self.assertEqual(chosen,pick(list(reversed(episodes)),3,1209,{'b'},{('a','0')})[0])

    def test_missing_groups_keep_denominator_and_bad_prefix_rejected(self):
        p = dict(models={'CONCAT_s42':{},'LOCAL_s42':{}},seeds=[42])
        entries = [dict(id=0,house='h'),dict(id=1,house='h')]
        outcomes = {a:dict(success=1,spl=.5,steps=10) for a in ('NATIVE',*p['models'])}
        result = summarize({0:dict(house='h',outcomes=outcomes)},entries,p)
        self.assertIsNone(result['arms']['CONCAT_s42']['sr'])
        self.assertEqual(result['arms']['CONCAT_s42']['identification_bounds'],[.5,1.])
        trace = [dict(event='reset',rgb_sha256='x'),dict(event='generation',environment_step=0,input={'rgb':'x'},native_logits=[0,1,0,0]),
                 dict(event='action',executed_action=1,before_rgb_sha256='x',after_rgb_sha256='y')]
        broken = copy.deepcopy(trace)
        broken[1]['input']['rgb']='wrong'
        with self.assertRaisesRegex(ValueError,'PROCESSED_PREFIX'):
            audit_pair(trace,broken,{'success':0},{'success':0})


if __name__=='__main__':
    unittest.main()
