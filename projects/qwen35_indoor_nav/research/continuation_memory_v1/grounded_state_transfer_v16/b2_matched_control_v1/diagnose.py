"""Final-head state and matched-memory diagnostics, outside training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
import objective as o


def main(run):
    import torch
    cfg=config(run);data=read(run/'DATA.json')
    cache={k:v.float().cuda() for k,v in torch.load(run/'features/FEATURES.pt',weights_only=True).items()}
    rows=[];interventions=[];seals={}
    with torch.inference_mode():
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                tag=f'{arm}_{seed}';net=o.initialize(seed).cuda().eval()
                path=run/'train'/tag/'FINAL.pt';result=read(path.parent/'RESULT.json')
                if sha(path)!=result['checkpoint_sha256']:raise ValueError('HEAD_CHANGED')
                net.load_state_dict(torch.load(path,weights_only=True));initial=c.model_identity(net)['sha256']
                for family in data['families']:
                    if family['split']!='DEV':continue
                    b=o.batch(family,'cuda');x=cache['features'][b['indices']];native=cache['logits'][b['indices']]
                    states,_=o.encode_masked(net,x,b['alive']);z=net.state_head(states.flatten(2)).sigmoid()
                    for i,r in enumerate(family['sequences']):
                        if not r['action_masks'][r['cutoff']]:continue
                        t=r['cutoff'];trained_bits=range(4) if arm=='B2' else (2,) if arm=='Terminal' else ()
                        rows.append(dict(model=tag,arm=arm,seed=seed,family=family['family_id'],parent=family['parent_family_id'],
                            stratum=family['stratum'],history=r['history'],task=r['task'],
                            trained_state_bits={str(k):dict(truth=r['state_targets'][t][k],probability=float(z[i,t,k])) for k in trained_bits}))
                    for case in family['mechanism_cases']:
                        i,t=case['correct'];j,tj=case['wrong'];k,tk=case['sham'];seq=family['sequences']
                        if len({seq[i]['features'][t],seq[j]['features'][tj],seq[k]['features'][tk]})!=1:raise ValueError('INTERVENTION_INPUT_CHANGED')
                        memories=dict(correct=states[i,t],wrong=states[j,tj],sham=states[k,tk],zero=torch.zeros_like(states[i,t]))
                        values={}
                        for mode,memory in memories.items():
                            logits=net.action_logits(memory.unsqueeze(0),native[i,t].unsqueeze(0),x[i,t].unsqueeze(0))[0]
                            values[mode]=dict(action=int(logits.argmax()),logits=logits.cpu().tolist())
                        interventions.append(dict(model=tag,arm=arm,seed=seed,parent=family['parent_family_id'],
                            stratum=family['stratum'],task=case['task'],target=case['target'],step=t,predictions=values))
                final=c.model_identity(net)['sha256']
                if initial!=final:raise ValueError('DIAGNOSTIC_PARAMETER_CHANGE')
                seals[tag]=final;del net
    immutable(run/'STATE_DIAGNOSIS.json',dict(rows=rows,scope='Only trained bits; DEV causal teacher histories, not autonomous success'))
    immutable(run/'MEMORY_INTERVENTIONS.json',dict(rows=interventions,scope='Read-only same-input actions; no extra closed-loop trajectories'))
    immutable(run/'DIAGNOSTICS_COMPLETE.json',dict(models=9,head_states=seals))


if __name__=='__main__':main(Path(sys.argv[1]))
