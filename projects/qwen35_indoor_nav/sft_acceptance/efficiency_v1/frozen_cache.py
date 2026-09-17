"""Bounded exact-input caches for the frozen preprocessing/vision stages only.

No language hidden state, recurrent memory, action labels, or teacher information
is cached. Install in a new runner, never by editing a sealed experiment.
"""
from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json

import torch
from PIL import Image


def fingerprint(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            assert not x.requires_grad, 'Cannot hash/cache trainable input'
            y = x.detach().contiguous().cpu()
            h.update(json.dumps(['tensor', str(y.dtype), str(x.device), list(y.shape)]).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, Image.Image):
            h.update(json.dumps(['PIL',x.mode,x.size]).encode());h.update(x.tobytes())
        elif isinstance(x, Mapping):
            h.update(b'mapping')
            for key in sorted(x):visit(key);visit(x[key])
        elif isinstance(x, (tuple,list)):
            h.update(json.dumps([type(x).__name__,len(x)]).encode())
            for item in x:visit(item)
        elif x is None or isinstance(x,(str,int,float,bool)):
            h.update(json.dumps([type(x).__name__,x],allow_nan=False).encode())
        else:raise TypeError(f'Unsupported cache key type {type(x)}')
    visit(value)
    return h.hexdigest()


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,Mapping):return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(x) for x in value)
    if value is None or isinstance(value,(str,int,float,bool)):return 0
    raise TypeError(f'Unsupported cached output {type(value)}')


def snapshot(value):
    """Clone returned containers too: BatchFeature.to mutates its container."""
    if isinstance(value,torch.Tensor):
        assert not value.requires_grad, 'Cannot detach a trainable path for caching'
        return value.detach().clone()
    if isinstance(value,Mapping):
        data={k:snapshot(v) for k,v in value.items()}
        if type(value) is dict:return data
        # HF BatchFeature is a UserDict; preserve its .to() interface.
        if hasattr(value,'data'):
            obj=copy.copy(value);obj.data=data;return obj
        # HF ModelOutput dataclasses accept named fields.
        return type(value)(**data)
    if isinstance(value,tuple):return tuple(snapshot(x) for x in value)
    if isinstance(value,list):return [snapshot(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(type(value))


class BoundedCache:
    def __init__(self,max_bytes=1024**3,max_entries=4096):
        assert max_bytes>0 and max_entries>0
        self.max_bytes=max_bytes;self.max_entries=max_entries
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0

    def clear(self):self.entries.clear();self.bytes=0

    def get_or_compute(self,key,compute):
        if key in self.entries:
            self.hits+=1
            value,size=self.entries.pop(key);self.entries[key]=(value,size)
            return snapshot(value)
        self.misses+=1
        value=compute();size=tensor_bytes(value)
        if size<=self.max_bytes:
            stored=snapshot(value)
            while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=self.max_entries):
                _,(_,old_size)=self.entries.popitem(last=False);self.bytes-=old_size
            self.entries[key]=(stored,size);self.bytes+=size
        return value

    def stats(self):return dict(hits=self.hits,misses=self.misses,tensor_bytes=self.bytes,entries=len(self.entries))


class CachedProcessor:
    """Processor config/tokenizer immutable for this wrapper's lifetime.

    Rebuild the wrapper on tokenizer, preprocessing or augmentation changes.
    Content keys include exact image pixels, text and all call options.
    """
    def __init__(self,processor,**limits):
        self.wrapped=processor;self.cache=BoundedCache(**limits)

    def __getattr__(self,key):return getattr(self.wrapped,key)

    def __call__(self,*args,**kwargs):
        key=fingerprint((args,kwargs))
        return self.cache.get_or_compute(key,lambda:self.wrapped(*args,**kwargs))


class CachedFrozenVision:
    """Cache only get_image_features while vision params AND buffers are frozen.

    Input device/dtype and visual module mode are part of the cache key.
    Caller must ensure stochastic vision operations are disabled; standard Q35N
    has no vision dropout. Any unfreeze is rejected, and in-place updates clear
    cached features. Tensor hashing can synchronize GPU; benchmark end-to-end.
    """
    def __init__(self,visual,encode,**limits):
        self.visual=visual;self.encode=encode;self.cache=BoundedCache(**limits);self.version=None

    def __call__(self,*args,**kwargs):
        assert all(not p.requires_grad for p in self.visual.parameters()), 'Vision is trainable; cache forbidden'
        for module in self.visual.modules():
            if module.training:
                assert not isinstance(module,torch.nn.modules.dropout._DropoutNd) or module.p==0, 'Stochastic vision'
                assert not getattr(module,'attention_dropout',0), 'Stochastic attention'
        tensors=list(self.visual.parameters())+list(self.visual.buffers())
        version=tuple((id(p),p._version,str(p.dtype),str(p.device)) for p in tensors)
        if version!=self.version:self.cache.clear();self.version=version
        modes=tuple(m.training for m in self.visual.modules())
        key=fingerprint((modes,args,kwargs))
        with torch.no_grad():return self.cache.get_or_compute(key,lambda:self.encode(*args,**kwargs))
