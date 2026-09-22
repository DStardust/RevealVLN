"""Read-only first-eight ordinary CHECK routes; cached actions are not closed-loop SR."""
import sys,json,hashlib,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent;B=R/'evidence_state_policy_v1'
sys.path.insert(0,str(B))
from common import c,read,sha
from model import EvidencePolicy
sys.path.insert(0,str(D))
from interface import route_logits
import torch

def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    plan=read(D/'AUDIT_PLAN.json');source=Path(plan['source_run']);data=read(R/'natural_transfer_v9/DATA.json')
    cachepath=Path(read(source/'BINDING.json')['feature_result']['ordinary_path'])
    cache={k:v.float() for k,v in torch.load(cachepath,map_location='cpu',weights_only=True).items()}
    results=[];began=time.monotonic()
    for tag,path in plan['heads'].items():
        p=Path(path);net=EvidencePolicy(1209,'MONOTONIC');net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));net.eval()
        before=c.model_identity(net)['sha256'];rows=[];bypass=True
        with torch.inference_mode():
            for index in plan['record_indices']:
                r=data['records'][index]
                if r['partition']!='check':raise ValueError('AUDIT_SPLIT')
                ids=torch.tensor(r['features']);base=cache['logits'][ids];features=cache['features'][ids];targets=torch.tensor(r['targets'])
                proposal=net(features[None],base[None],torch.ones(1,len(ids),dtype=torch.bool))['logits'][0]
                off=route_logits(base,proposal,torch.zeros(len(ids),dtype=torch.bool));bypass &= torch.equal(base,off)
                na=base.argmax(-1);ma=proposal.argmax(-1);vals=base.topk(2,-1).values;margin=vals[:,0]-vals[:,1]
                rows.append(dict(record=index,house=r['row']['scene_group'],decisions=len(ids),native_correct=int((na==targets).sum()),
                    method_correct=int((ma==targets).sum()),action_flips=int((na!=ma).sum()),lost_correct=int(((na==targets)&(ma!=targets)).sum()),
                    gained_correct=int(((na!=targets)&(ma==targets)).sum()),native_stop_to_continue=int(((na==3)&(ma!=3)).sum()),
                    native_continue_to_stop=int(((na!=3)&(ma==3)).sum()),mean_native_margin=float(margin.mean())))
        after=c.model_identity(net)['sha256']
        if before!=after:raise ValueError('PARAMETER_CHANGED')
        totals={k:sum(x[k] for x in rows) for k in ('decisions','native_correct','method_correct','action_flips','lost_correct','gained_correct','native_stop_to_continue','native_continue_to_stop')}
        results.append(dict(tag=tag,path=path,checkpoint_sha256=sha(p),parameters_unchanged=True,head_state_sha256=before,bypass_bitwise_equal=bypass,totals=totals,rows=rows))
    out=dict(status='CPU_READ_ONLY_CACHED_ACTION_AUDIT',scope=plan['scope'],heads=results,cache_path=str(cachepath),cache_sha256=sha(cachepath),
        data_sha256=sha(R/'natural_transfer_v9/DATA.json'),seconds=time.monotonic()-began,gpu_hours=0,optimizer_updates=0,new_navigation_episodes=0,
        learned_router_evaluated=False,ordinary_closed_loop_retention=None,note='Fixed source cache; no claim of cache/live parity or trajectory equivalence after action changes. No gate threshold fitting on this CHECK.')
    with (D/'OFFLINE_AUDIT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps([dict(tag=r['tag'],**r['totals'],bypass_bitwise_equal=r['bypass_bitwise_equal']) for r in results]))
if __name__=='__main__':main()
