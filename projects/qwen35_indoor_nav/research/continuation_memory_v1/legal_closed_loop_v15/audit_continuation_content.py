"""Read-only verification of the RGB/semantic bytes supporting sealed rollouts."""
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review as r


def main(run):
    run=run.resolve();assert run.parent==r.HERE
    protocol=r.c.read(r.HERE/'CONTINUATION_PROTOCOL.json')
    admitted,_=r.admitted_conditions([run],protocol)
    inspected={};observations=0;records=[];start=time.perf_counter()
    for rank,assignment in enumerate(protocol['rollouts']):
        if assignment['condition'] not in admitted:continue
        folder=run/'rollouts'/f'{rank:03d}'
        result=r.c.read(folder/'ROLLOUT.json');path=folder/'TRACE_PRIVILEGED.json'
        assert r.c.sha(path)==result['trace_sha256']
        trace=r.c.read(path)
        for observation in trace['observations']:
            observations+=1
            for kind in ('rgb','semantic'):
                digest=observation[kind+'_hash'];key=(kind,digest)
                if key not in inspected:
                    path=run/'content'/(digest+'.'+kind+'.npy')
                    assert path.is_file() and not path.is_symlink() and path.resolve().parent==run/'content'
                    array=np.load(path,allow_pickle=False)
                    assert hashlib.sha256(array.tobytes()).hexdigest()==digest
                    assert array.shape==((224,224,3) if kind=='rgb' else (224,224))
                    assert array.dtype==(np.uint8 if kind=='rgb' else np.uint32)
                    if kind=='semantic':
                        ids,counts=np.unique(array,return_counts=True)
                        inspected[key]={str(int(i)):int(n) for i,n in zip(ids,counts)}
                    else:inspected[key]=True
            assert observation['pixels']==inspected['semantic',observation['semantic_hash']]
        records.append(dict(rank=rank,trace_sha256=result['trace_sha256']))
    result=dict(run=run.name,sealed_conditions=sorted(admitted),rollouts=len(records),observations=observations,
        unique_arrays_rehashed=len(inspected),semantic_pixel_counts_recomputed=True,records=records,
        seconds=time.perf_counter()-start,passed=True,simulator_loaded=False,model_loaded=False,
        source_sha256=r.c.sha(Path(__file__)))
    r.c.write(r.HERE/(run.name+'_CONTENT_REVIEW.json'),result,True)
    print({k:v for k,v in result.items() if k!='records'})


if __name__=='__main__':main(Path(sys.argv[1]))
