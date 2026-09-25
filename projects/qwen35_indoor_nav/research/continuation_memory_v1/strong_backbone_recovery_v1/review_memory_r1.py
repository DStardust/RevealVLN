"""CPU-only review recovery: retain empty splits as unmeasured, never PASS."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from memory_v2 import ExecutionMemory


def review_split(pack,states):
    groups=pack['groups'];rows=[]
    for arm in ('NATIVE','BC','B2','OURS'):
        model=None
        if groups and arm!='NATIVE':
            model=ExecutionMemory(groups[0]['features'].shape[-1]).eval()
            model.load_state_dict(states[arm])
        correct=total=0;details=[]
        with torch.inference_mode():
            for g in groups:
                x=g['features'];nt,nh,ns,nf=x.shape;delta=torch.zeros_like(g['base_action_logits'])
                if model is not None:
                    out=model(x.reshape(nt*nh,ns,nf),g['lengths'].flatten(),g['actor_features'].reshape(nt*nh,ns,nf))
                    delta=out['delta'].reshape(nt,nh,ns,4)
                logits=g['base_action_logits']+delta
                valid=torch.arange(ns)[None,None,:]<g['lengths'][:,:,None]
                mask=g['action_known'] & valid
                if not bool(torch.isfinite(logits[mask]).all()):raise ValueError('NONFINITE_ACTION_LOGITS')
                pred=logits.argmax(-1)
                correct+=int((pred[mask]==g['action_targets'][mask]).sum());total+=int(mask.sum())
                for t in range(nt):
                    for h in range(nh):
                        step=int(g['lengths'][t,h]-1)
                        details.append(dict(family=g['family'],house=g['house'],task=g['tasks'][t],history=g['histories'][h],
                            continuations=g['continuations'],action=int(pred[t,h,step]),target=int(g['action_targets'][t,h,step]),
                            action_known=bool(mask[t,h,step]),native_action=int(g['base_action_logits'][t,h,step].argmax())))
        rows.append(dict(arm=arm,status='MEASURED' if total else 'NO_ADMITTED_SAMPLES',
            correct=correct if total else None,total=total,teacher_action_accuracy=correct/total if total else None,details=details))
    return dict(status='MEASURED' if any(r['total'] for r in rows) else 'NO_ADMITTED_SAMPLES',
        groups=len(groups),families=sorted({g['family'] for g in groups}),houses=sorted({g['house'] for g in groups}),arms=rows)


def main(run,output):
    run=run.resolve();output=output.resolve()
    output.mkdir(parents=True,exist_ok=False);began=time.time();torch.set_num_threads(4)
    # Verify completed evidence at the handoff boundary. No model load on GPU
    # and no training/feature/physical replay is invoked by this recovery.
    original_lock=u.read(run/'SOURCE_LOCK.json')
    for path,digest in original_lock.items():
        if u.sha(path)!=digest:raise ValueError('ORIGINAL_SOURCE_CHANGED:'+path)
    admission=u.read(run/'data/ADMISSION.json');packs={};states={};training={};binding=None
    for split in ('FIT','CHECK'):
        path=run/'data'/(split+'.pt')
        if u.sha(path)!=admission['feature_shas'][split]:raise ValueError('DATA_PACK_CHANGED:'+split)
        packs[split]=torch.load(path,map_location='cpu',weights_only=True)
        if packs[split]['split']!=split:raise ValueError('SPLIT_MISMATCH')
    schedule=u.read(run/'training/SCHEDULE_42.json')
    if len(schedule)!=1200:raise ValueError('SCHEDULE_INCOMPLETE')
    for arm in ('BC','B2','OURS'):
        folder=run/'training'/arm;receipt=u.read(folder/'RESULT.json');path=folder/'FINAL.pt'
        if u.sha(path)!=receipt['final_sha256']:raise ValueError('FINAL_WEIGHT_CHANGED:'+arm)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['step']!=1200 or saved['arm']!=arm or receipt['updates']!=1200:raise ValueError('TRAINING_INCOMPLETE:'+arm)
        if saved['binding']['base_updates']!=0 or saved['binding']['data_sha256']!=admission['feature_shas']['FIT']:raise ValueError('TRAINING_BINDING_MISMATCH')
        if binding is None:binding=saved['binding']
        elif binding!=saved['binding']:raise ValueError('UNMATCHED_TRAINING_BINDINGS')
        losses=[__import__('json').loads(line) for line in (folder/'STEPS.jsonl').read_text().splitlines()]
        if [r['step'] for r in losses]!=list(range(1,1201)) or [r['group'] for r in losses]!=schedule:raise ValueError('TRAINING_SCHEDULE_LOG_MISMATCH')
        states[arm]=saved['model'];training[arm]=dict(updates=1200,final_sha256=receipt['final_sha256'],last_loss=losses[-1]['loss'])
    results={split:review_split(pack,states) for split,pack in packs.items()}
    missing=[s for s,r in results.items() if r['status']!='MEASURED']
    exclusions={s:dict(Counter(a['excluded'] for a in admission['attempts'] if a['split'].upper()==s and 'excluded' in a)) for s in packs}
    result=dict(status='PARTIAL_ACTION_DIAGNOSTIC_CHECK_DATA_MISSING' if missing else 'FIXED_ACTION_DIAGNOSTIC_COMPLETE',
        results=results,training=training,missing_splits=missing,exclusions=exclusions,review_execution='COMPLETED',
        original_process_status=u.read(run/'STATUS.json'),original_failure_preserved=True,
        scope='Only admitted actual-fork action diagnostics; fitting is not closed-loop success or held-out generalization.',
        method_benefit_closed_loop='NOT_MEASURED',extra_training_updates=0,gpu_hours=0,
        wall_seconds=time.time()-began,source_sha256=u.sha(__file__))
    u.write(output/'ACTION_REVIEW.json',result)
    u.write(output/'STATUS.json',dict(status='TRAINING_COMPLETE_CHECK_DATA_MISSING' if missing else 'ACTION_DIAGNOSTIC_COMPLETE',
        phase='REVIEW_RECOVERED',training_complete=True,actual_updates=3600,review_execution_complete=True,
        evaluation_complete=not missing,missing_splits=missing,base_updates=0,extra_training_updates=0,gpu_hours=0,
        original_failure='WORKER_FAILED:review / IndexError on empty CHECK',closed_loop_method_benefit='NOT_MEASURED'))
    u.write(u.HERE/'REVIEW_RECOVERY.json',dict(run=run.name,path=str(output),status_path=str(output/'STATUS.json'),review_path=str(output/'ACTION_REVIEW.json')))
    print(__import__('json').dumps({s:[{k:r[k] for k in ('arm','status','correct','total','teacher_action_accuracy')} for r in result['results'][s]['arms']] for s in packs},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)
