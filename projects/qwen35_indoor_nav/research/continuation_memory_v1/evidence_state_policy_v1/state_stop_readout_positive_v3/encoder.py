"""Reuse the exact Qwen forward; raw arrays belong to this registered worktree."""
from functools import lru_cache
from shared import *
base=load('ident_frozen_encoder',V16/'encoder.py')
Forward=base.Forward
load_policy=base.load_policy
warmup=base.warmup

class RawStore:
    def __init__(self,data):self.data=data
    @lru_cache(maxsize=128)
    def pixels(self,ref):
        import numpy as np
        import hashlib
        info=self.data['contents'][ref];path=LINE/info['line_relative_path']
        if sha(path)!=info['file_sha256']:raise ValueError('RAW_ARRAY_CHANGED')
        a=np.load(path,allow_pickle=False)
        if a.shape!=(224,224,3) or a.dtype!=np.uint8 or hashlib.sha256(a.tobytes()).hexdigest()!=ref[7:]:raise ValueError('RAW_ARRAY_CONTRACT')
        return a
    def get(self,index,t):
        from PIL import Image
        row=self.data['features'][index]
        return dict(instruction=row['instruction'],images=[Image.fromarray(self.pixels(ref)) for ref in row['rgb_refs']],executed=row['executed'])
