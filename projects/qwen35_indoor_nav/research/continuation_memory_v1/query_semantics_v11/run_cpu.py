"""Explicit CPU-only lightweight training; no encoder/simulator process or fallback."""
import os
from pathlib import Path
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_DEVICE_BINDING_REQUIRED'
    config=c.read(HERE/'PROTOCOL.json')
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    for relative,digest in config['source_hashes'].items():
        assert c.sha(train.LINE/relative)==digest,'SOURCE_CHANGED:'+relative
    run=HERE/'run_001';run.mkdir(exist_ok=False)
    began=time.monotonic()
    try:
        caches={}
        identities={}
        for name,version in [('special','multifamily_v7'),('natural','natural_transfer_v9')]:
            folder=HERE.parent/version
            identity=c.read(folder/'run_001/FEATURE_RESULT.json')
            path=folder/'run_001/FEATURES.pt'
            assert c.sha(path)==identity['file_sha256']
            caches[name]=torch.load(path,map_location='cpu',weights_only=True)
            identities[name]=identity
        assert identities['special']['base_state_sha256']==identities['natural']['base_state_sha256']
        c.write(run/'RUNTIME_IDENTITY.json',dict(device='cpu',gpu_used=False,torch_version=torch.__version__,
            threads=4,encoder_updates=0,new_encoder_forwards=0,feature_identities=identities,
            protocol_sha256=c.sha(HERE/'PROTOCOL.json'),pid=os.getpid()),True)
        train.train(run,config,caches['special'],c.read(HERE.parent/'multifamily_v7/DATA.json'),
                    caches['natural'],c.read(HERE.parent/'natural_transfer_v9/DATA.json'))
        c.write(run/'LAUNCH_RESULT.json',dict(status='COMPLETE',cpu_wall_seconds=time.monotonic()-began,
            gpu_hours=0.,policy_head_optimizer_updates=len(config['seeds'])*len(config['arms'])*config['steps_per_arm'],
            base_optimizer_updates=0,foreign_processes_signaled=[]),True)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),cpu_wall_seconds=time.monotonic()-began),True)
        raise


if __name__=='__main__':main()
