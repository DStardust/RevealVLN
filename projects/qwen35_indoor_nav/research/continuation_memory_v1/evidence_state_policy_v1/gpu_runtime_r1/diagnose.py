"""Cached teacher-path diagnosis of every trained state mode, without selecting checkpoints."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    import torch
    cfg=config(run);data=read(run/'DATA.json')
    cache={k:v.float().cuda() for k,v in torch.load(run/'features/FEATURES.pt',map_location='cpu',weights_only=True).items()}
    for tag in os.environ['B2_TRAIN_TAGS'].split(','):
        path=run/'train'/tag/'FINAL.pt';record=read(path.parent/'RESULT.json')
        if sha(path)!=record['checkpoint_sha256']:raise ValueError('HEAD_IDENTITY')
        net=make_head(tag);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));net.cuda().eval()
        initial=c.model_identity(net)['sha256'];tables=[]
        with torch.inference_mode():
            for family in data['families']:
                # Auxiliary labels are read only after the policy's causal forward.
                rows=family['sequences'];maximum=max(len(r['features']) for r in rows)
                idx=torch.tensor([r['features']+[0]*(maximum-len(r['features'])) for r in rows],device='cuda')
                alive=torch.tensor([[True]*len(r['features'])+[False]*(maximum-len(r['features'])) for r in rows],device='cuda')
                out=net(cache['features'][idx],cache['logits'][idx],alive)
                row=dict(family=family['family_id'],parent=family['parent_family_id'],house=family['house'],split=family['split'],
                         actions=0,action_correct=0,native_correct=0,state_labels=[0]*4,state_correct=[0]*4,
                         event_labels=[0]*2,event_correct=[0]*2,state_correct_action_count=0,state_correct_action_correct=0)
                state=out['state'].cpu();events=out['event_logits'].sigmoid().cpu();actions=out['logits'].argmax(-1).cpu();native=cache['logits'][idx].argmax(-1).cpu()
                for i,r in enumerate(rows):
                    for t in range(len(r['features'])):
                        truth=r['state_targets'][t];correct=[int(state[i,t,k]>=.5)==truth[k] for k in range(4)]
                        for k in range(4):
                            row['state_labels'][k]+=r['state_masks'][t];row['state_correct'][k]+=int(correct[k])*r['state_masks'][t]
                        for k in range(2):
                            mask=r['event_masks'][t][k];row['event_labels'][k]+=mask;row['event_correct'][k]+=int(int(events[i,t,k]>=.5)==r['event_targets'][t][k])*mask
                        if r['action_masks'][t]:
                            right=int(actions[i,t])==r['targets'][t];row['actions']+=1;row['action_correct']+=int(right);row['native_correct']+=int(int(native[i,t])==r['targets'][t])
                            row['state_correct_action_count']+=int(all(correct));row['state_correct_action_correct']+=int(all(correct) and right)
                tables.append(row)
        if c.model_identity(net)['sha256']!=initial:raise ValueError('DIAG_CHANGED_HEAD')
        immutable(run/('DIAG_'+tag+'.json'),dict(model=tag,head_sha256=initial,rows=tables,checkpoint_selection=False,
                  note='Repeated teacher windows; match to registered teacher, not proof other actions are illegal. No closed-loop effect.'))

if __name__=='__main__':main(Path(sys.argv[1]))
