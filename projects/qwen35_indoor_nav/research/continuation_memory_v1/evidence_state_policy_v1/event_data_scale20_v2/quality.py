"""Physical admission and spatial deduplication; no model score is read."""
import hashlib
import math
import random
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def isolated_position(position, existing, distance):
    return all(math.dist(position,p)>=distance for p in existing)

def candidate_positions(backend,cfg,old):
    rng=random.Random(cfg['seed'])
    ids=sorted(set(i for values in backend.eligible.values() for i in values));rng.shuffle(ids)
    offsets=[(radius*math.cos(angle*math.pi/6),radius*math.sin(angle*math.pi/6))
             for radius in (.65,1.,1.5,2.,2.5) for angle in range(12)]
    # Round robin objects so early positions do not exhaust one semantic object.
    result=[]
    for dx,dz in offsets:
        for i in ids:
            center=backend.objects[i]['center']
            raw=backend.np.asarray([center[0]+dx,center[1],center[2]+dz],dtype=backend.np.float32)
            point=backend.sim.pathfinder.snap_point(raw).tolist()
            if not all(math.isfinite(v) for v in point):continue
            if math.hypot(point[0]-raw[0],point[2]-raw[2])>.5:continue
            if not isolated_position(point,old,cfg['parent_min_distance_m']):continue
            if not isolated_position(point,result,.5):continue
            result.append(point)
            if len(result)>=cfg['positions_per_house']:return result
    return result

def validate_trace(trace):
    actions=trace['actions'];obs=trace['observations']
    if not trace.get('complete') or trace.get('interior_state_assignments')!=0:raise ValueError('INCOMPLETE_OR_TELEPORTED_TRACE')
    if not actions or actions[-1]!='S' or 'S' in actions[:-1] or len(actions)>500:raise ValueError('STOP_OR_BUDGET')
    if len(obs)!=len(actions):raise ValueError('STOP_MUST_NOT_ADD_OBSERVATION')
    if trace['collisions']!=0:raise ValueError('COLLIDING_TRACE_NOT_TRAINING_LABEL')
    if any(not o.get('evidence_complete') for o in obs):raise ValueError('UNKNOWN_NOT_NEGATIVE')

def audit_family(family,checker,arrays):
    import numpy as np
    from evaluator_v16 import legacy
    if family['split']!='FIT':raise ValueError('NON_FIT_DATA')
    loaded={};manifest={};events=Counter();actions=Counter();seen_inputs=set();scales=Counter()
    compiler=legacy.Compiler(**family['compiler'])
    for key,ref in family['traces'].items():
        p=LINE/ref['path']
        if sha(p)!=ref['sha256']:raise ValueError('TRACE_SHA_CHANGED')
        trace=read(p);validate_trace(trace);loaded[key]=trace;actions.update(trace['actions'])
        atoms=compiler.atoms(trace['observations'])
        for t,obs in enumerate(trace['observations']):
            for kind,shape,dtype in (('rgb',(224,224,3),np.uint8),('semantic',(224,224),np.uint32)):
                h=obs[kind+'_hash'];path=LINE/family['content_root']/(h+'.'+kind+'.npy')
                if str(path) not in arrays:
                    a=np.load(path,allow_pickle=False)
                    if a.shape!=shape or a.dtype!=dtype or hashlib.sha256(a.tobytes()).hexdigest()!=h:raise ValueError('ARRAY_CONTRACT')
                    pixels=None
                    if kind=='semantic':
                        ids,counts=np.unique(a,return_counts=True)
                        pixels={str(int(i)):int(n) for i,n in zip(ids,counts)}
                    arrays[str(path)]=dict(sha256=sha(path),pixels=pixels)
                if kind=='semantic' and arrays[str(path)]['pixels']!=obs['pixels']:raise ValueError('PIXEL_COUNTS_CHANGED')
                manifest[str(path.relative_to(LINE))]=arrays[str(path)]['sha256']
            input_key=digest([o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]]+[trace['actions'][max(0,t-8):t]])
            for role in ('anchor','terminal'):
                k=(input_key,role)
                if k in seen_inputs:continue
                seen_inputs.add(k);events[role+('_positive' if atoms[t][role] else '_negative')]+=1
                prev=trace['observations'][t-1]['pixels'] if t else {}
                paired=max((min(prev.get(str(i),0),obs['pixels'].get(str(i),0)) for i in compiler.eligible[role]),default=0)
                scales[role+(' clear_ge1024' if paired>=1024 else ' threshold_256_1023' if paired>=256 else ' below_256')]+=1
    certificate=checker(compiler,family['histories'],family['suffixes'],loaded)
    if certificate!=family['certificate']:raise ValueError('CERTIFICATE_CHANGED')
    return dict(status='RAW_ARRAYS_AND_FROZEN_CERTIFICATE_VERIFIED',trace_count=len(loaded),
                crossed_labels=len(certificate['labels']),events_unique_within_family=dict(events),
                scale_unique_within_family=dict(scales),actions=dict(actions),arrays=manifest,
                warnings=['Stationary turns; forward-action coverage is zero.',
                          'Within-house parents and derived variants are correlated, not independent generalization samples.'])

def counts(run):
    families=[];houses={}
    for house in read(run/'PROTOCOL.json')['houses']:
        folder=run/'collect'/house
        ps=list(folder.glob('position_*/FAMILY.json'));qs=list(folder.glob('absent_*/FAMILY.json'))
        houses[house]=dict(parents=len(ps),counterparts=len(qs),complete=(folder/'DATASET.json').exists(),
                          progress=read(folder/'STATUS.json') if (folder/'STATUS.json').exists() else None)
        for p in ps+qs:
            f=read(p)
            if f.get('training_admission')!='CONTROLLED_SEE2_ARRAY_CERTIFIED_FIT':raise ValueError('UNADMITTED_FAMILY')
            families.append(f)
    parents={f['parent_family_id'] for f in families}
    if len({f['family_id'] for f in families})!=len(families):raise ValueError('DUPLICATE_FAMILY')
    return dict(new_parents=len(parents),new_variants=len(families),new_histories=4*len(families),
                certified_physical_cross_executions=12*len(families),crossed_labels=24*len(families),
                houses=houses,baseline_total_parents=40,baseline_FIT_parents=32,
                expansion_vs_old_total=len(parents)/40,expansion_vs_old_FIT=len(parents)/32,
                target_new_parents=800,planned_capacity=read(run/'PROTOCOL.json')['families_per_house']*len(houses)),families

