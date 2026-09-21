"""Read-only source bindings and diagnostic-only outputs."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
V16=HERE.parent
SOURCE=V16/'b2_matched_control_v1'
sys.path.insert(0,str(SOURCE))
import objective as o
from shared import c, read, write, sha, digest, immutable, atomic_torch, LINE
from evaluator_v16 import legacy,state_sequence
RUN=SOURCE/'runs/matched_001'
OUT=HERE/'run_002'


def metrics(truth,probability):
    pairs=list(zip(truth,probability));tp=tn=fp=fn=0
    for y,p in pairs:
        if y:tp+=p>=.5;fn+=p<.5
        else:fp+=p>=.5;tn+=p<.5
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    return dict(n=len(pairs),tp=tp,tn=tn,fp=fp,fn=fn,tpr=tpr,tnr=tnr,
                balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
                accuracy=(tp+tn)/len(pairs) if pairs else None)


def fit_weights(rows,labels):
    """One total weight per FIT parent, then equal positive/negative mass per bit."""
    from collections import Counter
    import torch
    if any(r['split']!='FIT' for r in rows):raise ValueError('NONFIT_PROBE_OPTIMIZATION')
    count=Counter(r['parent'] for r in rows)
    w=torch.tensor([1/count[r['parent']] for r in rows])[:,None].expand_as(labels).clone()
    for bit in range(labels.shape[1]):
        for value in (0,1):
            m=labels[:,bit]==value;mass=w[m,bit].sum()
            if not mass:raise ValueError('MISSING_FIT_CLASS')
            w[m,bit]/=2*mass
    return w/w.mean(0)


def age_bucket(age):
    return 'never' if age is None else 'current' if age==0 else '1-8' if age<=8 else '9-16' if age<=16 else '17+'


def merge_teacher(old,item):
    if old['state']!=item['state']:raise ValueError('CAUSAL_STATE_LABEL_CONFLICT')
    old['teacher_targets']=sorted(set(old['teacher_targets']+item['teacher_targets']))
    old['action_mask']=max(old['action_mask'],item['action_mask'])
    old['preservation']=old['preservation'] or item['preservation']
