"""Use the two immutable causal feature caches; no new base forward or update."""
from pathlib import Path
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


def main(run):
    config=c.read(HERE/'PROTOCOL.json')
    for path,digest in config['source_hashes'].items():
        assert c.sha(train.LINE/path)==digest,'SOURCE_CHANGED:'+path
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_IMMUTABLE_FEATURES'))
    special=HERE.parent/'multifamily_v7'
    natural=HERE.parent/'natural_transfer_v9'
    special_cache=torch.load(special/'run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    natural_cache=torch.load(natural/'run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    special_identity=c.read(special/'run_001/FEATURE_RESULT.json')
    natural_identity=c.read(natural/'run_001/FEATURE_RESULT.json')
    assert special_identity['base_state_sha256']==natural_identity['base_state_sha256']
    for cache in (special_cache,natural_cache):
        assert cache['features'].shape[0]==cache['logits'].shape[0]
        assert cache['features'].shape[1]==2048 and cache['logits'].shape[1]==4
        assert all(bool(torch.isfinite(value).all()) for value in cache.values())
    c.write(run/'FEATURE_REUSE.json',dict(special=special_identity,natural=natural_identity,
        new_encoder_forwards=0,encoder_updates=0,cache_recomputed=False),True)
    train.train(run,config,special_cache,c.read(special/'DATA.json'),natural_cache,c.read(natural/'DATA.json'))


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
