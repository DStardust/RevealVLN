"""Real frozen-head streaming plus denominator, incomplete-group and audit contracts."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry_value,admitted,prefix_audit
from review import metric,paired
from select_action import select

class Contracts(unittest.TestCase):
    def test_missing_families_do_not_shrink_denominator(self):
        cfg=read(HERE/'PROTOCOL.json');reg=registry_value([],cfg)
        self.assertEqual(len(reg['slots']),1536);self.assertEqual(reg['main_denominator_per_arm'],384)
        self.assertEqual(len(reg['models']),6);self.assertFalse(any(r['available'] for r in reg['conditions']))
        self.assertEqual(len({s['rank'] for s in reg['slots']}),1536)
        for i in range(6):self.assertEqual(reg['slots'][i*6]['model'],reg['models'][i])

    def test_unidentified_difference_bounds(self):
        rows=[dict(condition=0,seed=1209,arm='DIRECT',status='NOT_COLLECTED',cost=1.),
              dict(condition=0,seed=1209,arm='MONOTONIC',status='PASS',cost=.1)]
        p=paired(rows);self.assertEqual(p['delta_lower'],0);self.assertEqual(p['delta_upper'],1)
        rows[1]['status']='NOT_COLLECTED';p=paired(rows)
        self.assertEqual((p['delta_lower'],p['delta_upper']),(-1,1))

    def test_half_groups_not_admitted_and_bad_state_rejected(self):
        cfg=read(HERE/'PROTOCOL.json');reg=registry_value([],cfg)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);s=run/'evaluate/session_test';s.mkdir(parents=True)
            ranks=[s['rank'] for s in reg['slots'] if s['condition']==0]
            immutable(s/'GROUP_000.json',dict(condition=0,ranks=ranks,files={}))
            self.assertEqual(admitted(run,reg),{})
            write(s/'STATE_SEAL_000.json',dict(base_unchanged=False,heads_unchanged=True,completed_ranks=ranks),True)
            with self.assertRaisesRegex(ValueError,'STATE_CHANGED'):admitted(run,reg)

    def test_final_method_stop_semantics_and_nonfinite(self):
        self.assertEqual(select([0,0,0,5],[0,4,0,1])['executed_action'],'turn_left')
        self.assertEqual(select([3,2,1,0],[0,0,0,4])['executed_action'],'STOP')
        with self.assertRaises(ValueError):select([0,0,0,1],[0,0,float('nan'),1])

    def test_prefix_rejects_input_and_native_flip(self):
        a=dict(raw={'key':'a'},processed={'input':'x'},logits=[3.,2.,1.,0.],decision=0,**select([3.,2.,1.,0.],[3.,2.,1.,0.]))
        self.assertNotIn('first_divergence',prefix_audit([a],[dict(a)]))
        self.assertIn('first_divergence',prefix_audit([a],[dict(a,processed={'input':'changed'})]))
        b=dict(a,logits=[2.,3.,1.,0.],native_action='turn_left')
        self.assertEqual(prefix_audit([a],[b])['argmax_flip_count'],1)

    def test_loaded_final_heads_stream_without_updates(self):
        import torch
        torch.set_num_threads(2);cfg=read(HERE/'PROTOCOL.json')
        data=read(CPU/'DATA.json');cache=torch.load(CONTROL/'features/FEATURES.pt',map_location='cpu',weights_only=True)
        seq=data['families'][0]['sequences'][0];cut=seq['cutoff'];indices=seq['features'][:cut+1]
        x=cache['features'][indices].float().unsqueeze(0);y=cache['logits'][indices].float().unsqueeze(0)
        for arm in cfg['arms']:
            tag=arm+'_1209';net=make_head(tag);source=Path(cfg['source_models'])/'train'/tag
            net.load_state_dict(torch.load(source/'FINAL.pt',map_location='cpu',weights_only=True));net.eval()
            before=c.model_identity(net)['sha256'];self.assertEqual(before,read(source/'RESULT.json')['final'])
            with torch.inference_mode():
                state=prefill(net,[x[:,t] for t in range(cut)])
                logits,state,detail=net.step(x[:,cut],y[:,cut],state)
                full=net(x,y,torch.ones(1,cut+1,dtype=torch.bool))
                self.assertTrue(torch.equal(logits,full['logits'][:,-1]));self.assertTrue(torch.equal(detail['state'],full['state'][:,-1]))
                self.assertEqual(set(state_identity(state)),{'memory','belief','first'})
            self.assertEqual(before,c.model_identity(net)['sha256'])

    def test_collector_uses_existing_physical_certifiers(self):
        collect=load('holdout_test_collector',HERE/'collect.py')
        self.assertEqual(collect.present.HISTORIES,collect.absent.HISTORIES)
        self.assertEqual(set(collect.present.QUERIES),{'direct_stop','acquire_anchor','wrong_terminal'})
        self.assertEqual(set(collect.absent.QUERIES),{'direct_stop','acquire_anchor','return_terminal'})

    def test_cleanup_refuses_foreign_owner(self):
        import subprocess
        m=load('holdout_test_ownership',V16/'pipeline.py')
        proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        try:
            who=m.process_identity(proc.pid)
            with self.assertRaises(Exception):m.cleanup(proc,dict(who,uid=who['uid']+1))
            self.assertIsNone(proc.poll())
        finally:proc.terminate();proc.wait(timeout=10)

if __name__=='__main__':unittest.main()
