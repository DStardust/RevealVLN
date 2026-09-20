"""Actual-feature CPU checks for ragged histories and late-loss gradient paths."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import objective as o


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=o.c.read(HERE/'DATA.json');family=data['families'][0]
    part=HERE/'features_run_001/PART_002048.pt'
    cache={k:v.float() for k,v in torch.load(part,map_location='cpu',weights_only=True).items()}
    b=o.batch(family);assert int(b['indices'].max())<len(cache['features'])
    assert len(set(b['lengths'].tolist()))>1
    net=o.initialize(1209);features=cache['features'][b['indices']]
    with torch.no_grad():
        all_states,_=net.encode(features)
        for i,n in enumerate(b['lengths'].tolist()):
            one,_=net.encode(features[i:i+1,:n])
            assert torch.allclose(one[0,-1],all_states[i,n-1],atol=1e-6,rtol=1e-6)
        memory=all_states[torch.arange(len(b['lengths'])),b['lengths']-1]
        before=memory.clone();current=cache['features'][b['indices'][torch.arange(len(b['lengths'])),b['lengths']-1]]
        action_before=net.action_logits(memory,torch.zeros(len(memory),4),current)
        net.reader(memory,torch.zeros(len(memory),4));net.reader(memory,torch.ones(len(memory),4))
        assert torch.equal(memory,before)
        assert torch.equal(action_before,net.action_logits(memory,torch.zeros(len(memory),4),current))
    positive=next(i for i,p in enumerate(family['prefixes']) if p['history_id']=='H_A' and p['task_id']=='task_A')
    prefix=family['prefixes'][positive];critical=next(t for t,z in enumerate(prefix['state_targets']) if z[1]);gap=len(prefix['features'])-1-critical
    assert gap>8
    rows=[]
    for arm in ('B1','B2','Ours'):
        net=o.initialize(1209);before={k:v.clone() for k,v in net.state_dict().items()}
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);loss,_,_=o.losses(net,cache,b,arm)
            assert bool(torch.isfinite(loss));loss.backward();optimizer.step()
        _,_,parts=o.losses(net,cache,b,arm,retain_steps=[critical])
        if arm=='Ours':
            ids=[i for i,cell in enumerate(family['cells']) if cell['prefix']==positive]
            late=F.binary_cross_entropy_with_logits(parts['query_logits'][ids],torch.tensor([family['cells'][i]['y'] for i in ids],dtype=torch.float32))
        elif arm=='B2':
            late=F.binary_cross_entropy_with_logits(net.state_head(parts['final'][positive:positive+1].flatten(1)),torch.tensor([prefix['state_targets'][-1]],dtype=torch.float32))
        else:
            case=next(x for x in b['forks'] if x['left']==positive)
            late=o.fork_logits(net,cache,b,parts['final'],case).square().sum()
        gradient=torch.autograd.grad(late,parts['writes'][critical])[0][positive]
        assert bool(torch.isfinite(gradient).all()) and float(gradient.norm())>0
        changed={k:not torch.equal(before[k],v) for k,v in net.state_dict().items()}
        assert changed['writer.weight'] and changed['recurrent.weight'] and changed['action.2.weight']
        rows.append(dict(arm=arm,disposable_updates=2,late_loss_writer_gradient_norm=float(gradient.norm()),
            writer_changed=True,recurrent_changed=True,action_changed=True))
    o.c.write(HERE/'CPU_TEST_RESULT.json',dict(passed=True,actual_new_qwen_feature_rows=len(cache['features']),
        feature_part_sha256=o.c.sha(part),ragged_final_state_check=True,future_query_cannot_change_memory_or_action=True,
        critical_step=critical,supervision_step=len(prefix['features'])-1,event_gap=gap,
        actual_disposable_optimizer_updates=6,runs=rows,gpu_hours=0,formal_method_comparison=False,
        encoder_final_fingerprint_required_before_formal_training=True),True)
    print(rows,flush=True)


if __name__=='__main__':main()
