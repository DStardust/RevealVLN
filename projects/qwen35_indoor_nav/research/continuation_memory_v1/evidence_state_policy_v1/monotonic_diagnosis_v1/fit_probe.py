"""Frozen CPU forward at every existing FIT teacher takeover, no updates or new labels."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import HERE,BASE,HOLDOUT,CPU,read,sha,immutable,c,confusion
from shared import make_head,CONTROL

def main(output):
    import torch
    torch.set_num_threads(2);began=time.monotonic()
    data=read(CPU/'DATA.json');cache_path=CONTROL/'features/FEATURES.pt';cache_record=read(CONTROL/'features/FEATURE_RESULT.json')
    if sha(cache_path)!=cache_record['file_sha256']:raise ValueError('CACHE_CHANGED')
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    fit=[f for f in data['families'] if f['split']=='FIT'];results=[];records=[]
    for seed in (1209,1210,1211):
        for arm in ('DIRECT','MONOTONIC'):
            tag=f'{arm}_{seed}';folder=HOLDOUT/'runs/holdout_001/train'/tag;record=read(folder/'RESULT.json')
            if sha(folder/'FINAL.pt')!=record['checkpoint_sha256']:raise ValueError('FINAL_HEAD_CHANGED')
            net=make_head(tag);net.load_state_dict(torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True));net.eval()
            before=c.model_identity(net)['sha256']
            if before!=record['final']:raise ValueError('HEAD_IDENTITY')
            with torch.inference_mode():
                for family in fit:
                    rows=[r for r in family['sequences'] if any(r['action_masks'])]
                    if len(rows)!=8 or any(not r['action_masks'][r['cutoff']] for r in rows):raise ValueError('TEACHER_CUTOFF_SELECTION')
                    lengths=[r['cutoff']+1 for r in rows];maximum=max(lengths)
                    ids=torch.tensor([r['features'][:n]+[0]*(maximum-n) for r,n in zip(rows,lengths)])
                    alive=torch.tensor([[True]*n+[False]*(maximum-n) for n in lengths])
                    out=net(cache['features'][ids].float(),cache['logits'][ids].float(),alive)
                    for i,r in enumerate(rows):
                        t=r['cutoff'];z=out['state'][i,t];truth=torch.tensor(r['state_targets'][t],dtype=z.dtype);logits=out['logits'][i,t]
                        exact=logits+net.state_action(truth-z)
                        records.append(dict(model=tag,arm=arm,seed=seed,family=family['family_id'],house=family['house'],task=r['task'],history=r['history'],stratum=family['stratum'],
                            target=r['targets'][t],action=int(logits.argmax()),native_action=int(cache['logits'][r['features'][t]].argmax()),
                            state_correct=bool(((z>=.5)==truth.bool()).all()),truth=truth.tolist(),predicted=z.tolist(),
                            exact_state_algebra_action=int(exact.argmax()),stop_margin=float(logits[3]-logits[:3].max())))
            after=c.model_identity(net)['sha256']
            if before!=after:raise ValueError('CPU_PROBE_UPDATED_MODEL')
            for task in ('task_A','task_T'):
                selected=[r for r in records if r['model']==tag and r['task']==task];ready=[r for r in selected if r['truth'][3]];correct=[r for r in selected if r['state_correct']]
                results.append(dict(model=tag,task=task,n=len(selected),teacher_action_correct=sum(r['action']==r['target'] for r in selected),
                    state_correct=len(correct),state_correct_action_wrong=sum(r['action']!=r['target'] for r in correct),
                    ready_n=len(ready),ready_stop=sum(r['action']==3 for r in ready),
                    ready_predicted_continue=sum(r['predicted'][3]>=.5 and r['action']!=3 for r in ready),
                    ready_exact_state_algebra_stop=sum(r['exact_state_algebra_action']==3 for r in ready),
                    ready_actions=dict(Counter(r['action'] for r in ready))))
            print(tag,'FIT takeover checked',flush=True)
    immutable(output,dict(status='FROZEN_FIT_CPU_FORWARD_COMPLETE',models=6,records=records,summary=results,
        model_state_unchanged=True,new_updates=0,gpu_hours=0,new_env_actions=0,seconds=time.monotonic()-began,
        source_sha256=sha(Path(__file__)),data_sha256=sha(CPU/'DATA.json'),cache_sha256=cache_record['file_sha256'],
        scope='All 59 existing FIT variants, eight real teacher takeover sequences each; six frozen final heads. CPU features only, not Qwen rerun or navigation. Exact-state algebra is diagnostic, never executed.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
