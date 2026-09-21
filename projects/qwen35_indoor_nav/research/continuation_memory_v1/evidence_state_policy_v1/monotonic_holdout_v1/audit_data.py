"""Independently reread arrays and certificates before sealing the score-blind registry."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry_value

def main(run):
    import numpy as np
    collector=load('holdout_audit_collector',HERE/'collect.py')
    present,absent=collector.present,collector.absent
    from evaluator_v16 import legacy
    cfg=config(run);families=[];arrays={};traces_count=0;attempts=0
    for house in cfg['houses']:
        path=run/'collect'/house/'DATASET.json'
        families.extend(read(path)['families'])
        attempts+=len(c.records(path.parent/'ATTEMPTS.jsonl'))
    parents=set();old_houses={f['house'] for f in read(run/'WARMUP_DATA.json')['raw_families']}
    for family in families:
        if family['split']!='TEST' or family['house'] in old_houses or family['training_admission'] is not False:raise ValueError('DATA_ISOLATION')
        loaded={};parents.add(family['parent_family_id'])
        for key,ref in family['traces'].items():
            path=LINE/ref['path']
            if sha(path)!=ref['sha256']:raise ValueError('TRACE_SHA_CHANGED')
            trace=read(path);loaded[key]=trace;traces_count+=1
            for obs in trace['observations']:
                for kind,shape,dtype in (('rgb',(224,224,3),np.uint8),('semantic',(224,224),np.uint32)):
                    digest_=obs[kind+'_hash'];path=LINE/family['content_root']/(digest_+'.'+kind+'.npy')
                    if str(path) not in arrays:
                        a=np.load(path,allow_pickle=False)
                        if a.shape!=shape or a.dtype!=dtype or hashlib.sha256(a.tobytes()).hexdigest()!=digest_:raise ValueError('ARRAY_CONTRACT')
                        pixels=None
                        if kind=='semantic':
                            ids,counts=np.unique(a,return_counts=True);pixels={str(int(i)):int(n) for i,n in zip(ids,counts)}
                        arrays[str(path)]=dict(sha256=sha(path),pixels=pixels)
                    if kind=='semantic' and arrays[str(path)]['pixels']!=obs['pixels']:raise ValueError('PIXEL_COUNTS_CHANGED')
        checker=present.certify if family['stratum']=='terminal_present' else absent.certify
        if checker(legacy.Compiler(**family['compiler']),family['histories'],family['suffixes'],loaded)!=family['certificate']:raise ValueError('CERTIFICATE_CHANGED')
    data=dict(raw_families=families,scope=cfg['scope'],training_admission=False)
    immutable(run/'DATA.json',data)
    reg=registry_value(families,cfg);immutable(run/'EVALUATION_REGISTRY.json',reg)
    immutable(run/'DATA_AUDIT.json',dict(status='RAW_ARRAYS_AND_CERTIFICATES_VERIFIED',parents=len(parents),variants=len(families),planned_variants=cfg['planned_variants'],
        traces=traces_count,crossed_labels=len(families)*24,independent_houses=len({f['house'] for f in families}),
        arrays=len(arrays),attempt_ledger_rows=attempts,planned_slots=len(reg['slots']),available_slots=sum(reg['conditions'][s['condition']]['available'] for s in reg['slots']),
        data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json'),training_admission=False))
    immutable(run/'ARRAY_MANIFEST.json',{p:v['sha256'] for p,v in arrays.items()})
    print(read(run/'DATA_AUDIT.json'),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
