import importlib.util
from pathlib import Path
import unittest
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPooling

s=importlib.util.spec_from_file_location('frozen_cache',Path(__file__).with_name('frozen_cache.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class CacheTests(unittest.TestCase):
    def test_content_not_identity(self):
        self.assertEqual(m.fingerprint(Image.new('RGB',(2,2),'red')),m.fingerprint(Image.new('RGB',(2,2),'red')))
        self.assertNotEqual(m.fingerprint(Image.new('RGB',(2,2),'red')),m.fingerprint(Image.new('RGB',(2,2),'blue')))
    def test_text_order_dtype_keys(self):
        self.assertNotEqual(m.fingerprint(['a','b']),m.fingerprint(['b','a']))
        self.assertNotEqual(m.fingerprint(torch.ones(2)),m.fingerprint(torch.ones(2,dtype=torch.bfloat16)))
    def test_mutation_isolation(self):
        c=m.BoundedCache();v=c.get_or_compute('a',lambda:{'x':torch.ones(2)})
        v['x'].zero_();self.assertTrue(torch.equal(c.get_or_compute('a',lambda:None)['x'],torch.ones(2)))
    def test_hit_no_compute(self):
        c=m.BoundedCache();c.get_or_compute('a',lambda:torch.ones(2))
        c.get_or_compute('a',lambda:(_ for _ in ()).throw(RuntimeError()))
        self.assertEqual(c.hits,1)
    def test_memory_bound(self):
        c=m.BoundedCache(max_bytes=8,max_entries=1)
        for i in range(10):c.get_or_compute(str(i),lambda:torch.ones(2))
        self.assertEqual(c.bytes,8);self.assertEqual(len(c.entries),1)
    def test_large_bypass(self):
        c=m.BoundedCache(max_bytes=1);c.get_or_compute('a',lambda:torch.ones(2));self.assertEqual(c.bytes,0)
    def test_gradient_not_detached(self):
        with self.assertRaises(AssertionError):m.snapshot(torch.ones(2,requires_grad=True))
    def test_batchfeature(self):
        b=BatchFeature({'input_ids':torch.ones(1,2,dtype=torch.long)})
        c=m.snapshot(b);self.assertIsInstance(c,BatchFeature)
        c.data['input_ids'].zero_();self.assertEqual(int(b['input_ids'].sum()),2)
    def test_modeloutput(self):
        o=BaseModelOutputWithPooling(last_hidden_state=torch.ones(1,2),pooler_output=(torch.ones(1,2),))
        c=m.snapshot(o);self.assertIsInstance(c,BaseModelOutputWithPooling)
        c.pooler_output[0].zero_();self.assertEqual(float(o.pooler_output[0].sum()),2)
    def test_frozen_invalidation_and_unfreeze(self):
        v=torch.nn.Linear(2,2,bias=False);v.requires_grad_(False)
        f=m.CachedFrozenVision(v,v);x=torch.ones(1,2)
        a=f(x);self.assertTrue(torch.equal(a,f(x)));self.assertEqual(f.cache.hits,1)
        with torch.no_grad():v.weight.add_(1)
        self.assertFalse(torch.equal(a,f(x)))
        v.requires_grad_(True)
        with self.assertRaises(AssertionError):f(x)
    def test_stochastic_rejected(self):
        v=torch.nn.Dropout(.1);v.train();f=m.CachedFrozenVision(v,v)
        with self.assertRaises(AssertionError):f(torch.ones(2))
    def test_processor_causal_key(self):
        calls=[]
        def p(**kw):calls.append(kw);return BatchFeature({'input_ids':torch.tensor([len(kw['text'])])})
        c=m.CachedProcessor(p)
        c(text='a');c(text='a');c(text='b')
        self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
