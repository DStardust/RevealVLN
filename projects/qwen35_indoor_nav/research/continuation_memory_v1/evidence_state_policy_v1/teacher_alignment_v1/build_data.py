"""Use certified action alternatives; common original causal auxiliary pool stays intact."""
import copy
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from prepare import first_ready,supervision_audit

def apply_overlay(data,overlays):
    result=copy.deepcopy(data);families={f['family_id']:f for f in result['families']}
    for item in overlays:
        family=families[item['family_id']]
        if family['split']!='FIT':raise ValueError('NONFIT_OVERLAY')
        row=family['sequences'][item['sequence']];t=item['stop_step']
        if t!=first_ready(row) or not row['action_masks'][t]:raise ValueError('UNREGISTERED_STOP_POINT')
        row['targets'][t]=3;row['action_masks'][t+1:]=[0]*(len(row['targets'])-t-1)
        row['alternative_teacher_certificate']=item['plan']
    return result

def main(run):
    import numpy as np
    from collect import validate_prefix
    from evaluator_v16 import legacy,evaluate
    cfg=config(run);plan=read(run/'TEACHER_PLAN.json');data=read(run/'DATA.json')
    families={f['family_id']:f for f in data['raw_families']};verified=set();files={}
    for item in plan['plans']:
        path=run/'certificates'/item['id']/'CERTIFICATE.json';cert=read(path);files[str(path)]=sha(path)
        tracepath=LINE/cert['trace']['path'];files[str(tracepath)]=sha(tracepath)
        if cert['plan_sha256']!=digest(item) or files[str(tracepath)]!=cert['trace']['sha256']:raise ValueError('TEACHER_SEAL_CHANGED')
        trace=read(tracepath);ref=LINE/item['source_trace']['path']
        if sha(ref)!=item['source_trace']['sha256']:raise ValueError('ORIGINAL_TRACE_CHANGED')
        validate_prefix(trace,read(ref),item['stop_step'])
        compiler=legacy.Compiler(**families[item['family_id']]['compiler'])
        for task in item['tasks']:
            if evaluate(compiler,trace,task['task'],task['cutoff'])['safe_v16_label']!='PASS':raise ValueError('LABEL_RECOMPUTATION')
        for obs in trace['observations']:
            for kind,shape,dtype in [('rgb',(224,224,3),np.uint8),('semantic',(224,224),np.uint32)]:
                p=LINE/cert['content_root']/(obs[kind+'_hash']+'.'+kind+'.npy')
                if p in verified:continue
                a=np.load(p,allow_pickle=False)
                if a.shape!=shape or a.dtype!=dtype or hashlib.sha256(a.tobytes()).hexdigest()!=obs[kind+'_hash']:raise ValueError('PHYSICAL_ARRAY_CORRUPTION')
                files[str(p)]=sha(p);verified.add(p)
        # Reuse cached input only because instruction, raw RGB window, and actually executed
        # motion prefix match each recorded policy decision (including the STOP input).
        for overlay in [o for o in plan['overlays'] if o['plan']==item['id']]:
            f=next(f for f in data['families'] if f['family_id']==overlay['family_id']);r=f['sequences'][overlay['sequence']]
            raw=families[f['family_id']];instruction=raw['terminal_instruction'] if r['task']=='task_T' else compiler.tasks[r['task']]['instruction']
            for t in range(item['stop_step']+1):
                feature=data['features'][r['features'][t]]
                rgb=['sha256:'+o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]]
                executed=[c.ACTIONS['FLR'.index(a)] for a in trace['actions'][max(0,t-8):t]]
                if (feature['instruction'],feature['rgb_refs'],feature['executed'])!=(instruction,rgb,executed):raise ValueError('CERTIFIED_CACHE_INPUT_MISMATCH')
    aligned=apply_overlay(data,plan['overlays']);immutable(run/'DATA_ALIGNED.json',aligned)
    immutable(run/'ALIGNED_SUPERVISION_AUDIT.json',supervision_audit(aligned))
    for name in ['DATA.json','DATA_ALIGNED.json','TEACHER_PLAN.json','SCHEDULES.json','ORDINARY_DATA.json','PROTOCOL.json','SOURCE_LOCK.json','features/FEATURES.pt','features/ORDINARY_FEATURES.pt']:
        p=run/name;files[str(p)]=sha(p)
    immutable(run/'BINDING.json',dict(files=files,feature_result=read(CPU/'BINDING.json')['feature_result']))
    immutable(run/'BUILD_RESULT.json',dict(certified_executions=len(plan['plans']),changed_teacher_rows=len(plan['overlays']),
        raw_arrays_verified=len(verified),original=supervision_audit(data)['counts'],aligned=supervision_audit(aligned)['counts'],
        common_auxiliary_pool=True,dev_unchanged=[f for f in aligned['families'] if f['split']=='DEV']==[f for f in data['families'] if f['split']=='DEV'],
        base_forwards=0,cache_inputs_physically_matched=True,teacher_stop_is_actually_executed=True))
    print(read(run/'BUILD_RESULT.json'),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
