"""Architecture contracts, including a real TRAIN-cache backward smoke test."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import EvidenceMemory
import common as u
import torch
from torch import nn
import torch.nn.functional as F


class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def trained(self):
        m=EvidenceMemory(16)
        with torch.no_grad():m.actor[-1].weight.normal_(0,.1)
        return m
    def test_empty_memory_exactly_cancels_after_learning(self):
        m=self.trained();x=torch.randn(2,16)
        self.assertTrue(torch.equal(m.action_delta(x,m.reset(2)),torch.zeros(2,4)))
    def test_query_only_bias_cancels(self):
        class QueryOnly(nn.Module):
            def forward(self,x):return x[:,:4]*13+7
        m=EvidenceMemory(16);m.actor=QueryOnly()
        self.assertTrue(torch.equal(m.action_delta(torch.randn(2,16),torch.randn(2,8,64)),torch.zeros(2,4)))
    def test_early_event_has_real_gradient(self):
        m=self.trained();x=torch.randn(30,16,requires_grad=True);memory=m.reset()
        for step in range(len(x)):memory=m.update(x[step:step+1],memory)
        m.action_delta(torch.randn(1,16),memory).square().sum().backward()
        self.assertGreater(float(x.grad[0].abs().sum()),0)
        self.assertGreater(float(m.writer.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.query.weight.grad.abs().sum()),0)
    def test_rejection_exactly_preserves_native(self):
        m=self.trained();native=torch.randn(2,4)
        result=m.policy_logits(torch.randn(2,16),torch.randn(2,8,64),native)
        self.assertFalse(bool(result['accept'].any()));self.assertTrue(torch.equal(result['logits'],native))
    def test_accepted_correction_is_nonzero(self):
        m=self.trained()
        with torch.no_grad():m.intervention[-1].bias[1]=5
        native=torch.randn(2,4);r=m.policy_logits(torch.randn(2,16),torch.randn(2,8,64),native)
        self.assertTrue(bool(r['accept'].all()));self.assertGreater(float(r['delta'].detach().abs().sum()),0)
        self.assertTrue(torch.allclose(r['memory_attention'].sum(-1),torch.ones(2)))
    def test_episode_reset_is_clean(self):
        m=EvidenceMemory(16);changed=m.update(torch.randn(1,16),m.reset())
        self.assertGreater(float(changed.detach().abs().sum()),0);self.assertEqual(float(m.reset().abs().sum()),0)


def real_train_probe():
    parent=u.HERE/'preservation_runs/preserve_001'
    # This is a native public-backbone TRAIN capture, not a memory-head rollout
    # or a teacher trained on the old val_unseen mechanism family.
    caches=sorted(parent.glob('capture/evaluation/*/episodes/*/CACHE.pt'))
    row=None
    for path in caches:
        candidate=torch.load(path,map_location='cpu',weights_only=True)
        if candidate['partition']=='FIT':row=candidate;break
    assert row is not None and row['source']=='ACTUAL_FROZEN_NATIVE_ROLLOUT_NO_SUCCESS_FILTER'
    assert row['action_boundary']=='AFTER_NATIVE_ASSISTANT_HEADER_V2'
    train_source=u.ASSETS/'third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz'
    with gzip.open(train_source,'rt') as stream:episodes=json.load(stream)['episodes']
    assert any(str(e['episode_id'])==str(row['episode_id']) and Path(e['scene_id']).stem==row['house'] for e in episodes)
    torch.manual_seed(42);model=EvidenceMemory(row['memory_features'].shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    initial=model.actor[-1].weight.detach().clone();losses=[];gradient=[]
    query={int(step):i for i,step in enumerate(row['query_steps'])}
    for _ in range(2):
        memory=model.reset();logits=[]
        for step,x in enumerate(row['memory_features']):
            memory=model.update(x[None],memory)
            if step in query:
                i=query[step];logits.append(row['base_logits'][i]+model.action_delta(row['actor_features'][i:i+1],memory)[0])
        loss=F.cross_entropy(torch.stack(logits),row['targets']);optimizer.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        gradient.append(float(model.writer.weight.grad.norm()));losses.append(float(loss.detach()));optimizer.step()
    assert gradient[-1]>0 and not torch.equal(initial,model.actor[-1].weight)
    return dict(cache=str(path),cache_sha256=u.sha(path),episode_id=row['episode_id'],house=row['house'],partition='OFFICIAL_TRAIN/FIT',
        train_source=str(train_source),train_source_sha256=u.sha(train_source),
        feature_width=row['memory_features'].shape[-1],causal_steps=len(row['memory_features']),queries=len(query),
        cpu_debug_updates=2,losses=losses,writer_gradient_norms=gradient,parameters_changed=True,
        total_parameters=sum(p.numel() for p in model.parameters()),base_model_loaded=False,gpu_hours=0,
        trained_weights_saved=False,scope='Forward/backward/update engineering smoke, not method training or efficacy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    probe=real_train_probe() if result.wasSuccessful() else None
    u.write(a.output,dict(status='PASSED' if result.wasSuccessful() else 'FAILED',contract_tests=result.testsRun,
        failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],real_train_cache_probe=probe,
        research_status='UNTESTED',live_integration='NOT_IMPLEMENTED',sr=None,
        source={str(path):u.sha(path) for path in [Path(__file__).with_name('model.py'),Path(__file__),u.HERE/'memory_v2.py',u.HERE/'common.py']}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
