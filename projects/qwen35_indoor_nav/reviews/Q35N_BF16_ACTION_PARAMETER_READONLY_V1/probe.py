"""Read-only numeric representability probe; not a training update or navigation test."""
import hashlib,json,math,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
A=LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt'
B=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6r1/formal/attempt_001/checkpoint_000001000.pt'
C=LINE/'sft_acceptance/ordinary_onpolicy_adapt_v6/formal/attempt_001/checkpoint_000000950.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (HERE/'RESULT.json').exists()
    import torch
    assert not torch.cuda.is_initialized()
    assert sha(A)=='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    assert sha(B)=='d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a'
    assert sha(C)=='a00a6006bdea3ec0d30b79cfad538deaf44242f9c2e972c1f43207ca43433cda'
    a=torch.load(A,map_location='cpu',weights_only=True);b=torch.load(B,map_location='cpu',weights_only=True);c=torch.load(C,map_location='cpu',weights_only=True)
    groups=b['optimizer']['param_groups'];assert len(groups)==1
    g=groups[0];beta1,beta2=g['betas'];rows=[]
    for name in ('action_query','exec_embed.weight'):
        p=b['trainable'][name];assert p.dtype==torch.bfloat16
        slots=[s for s in b['optimizer']['state'].values() if s['exp_avg'].shape==p.shape]
        assert len(slots)==1,'AMBIGUOUS_PARAMETER_TO_MOMENT_MAPPING'
        slot=slots[0];step=int(slot['step']);assert step==5000
        # Frozen saved moments at their saved step. No new gradient; NOT an AdamW execution replay.
        x=p.double();m=slot['exp_avg'].double();v=slot['exp_avg_sq'].double()
        intended=x*(1-g['lr']*g['weight_decay'])-(g['lr']/(1-beta1**step))*m/(v.sqrt()/math.sqrt(1-beta2**step)+g['eps'])
        assert torch.isfinite(intended).all()
        change=(intended-x).abs();nonzero=change>0
        bf=intended.to(torch.bfloat16).double();fp=intended.to(torch.float32).double()
        rows.append(dict(name=name,elements=p.numel(),parameter_dtype=str(p.dtype),moment_dtype=str(slot['exp_avg'].dtype),
          learning_rate=g['lr'],saved_optimizer_step=step,
          actual_changed_coordinates_4000_to_corrected1000=int((p!=a['trainable'][name]).sum()),
          actual_changed_coordinates_950_to1000=int((p!=c['trainable'][name]).sum()),
          static_intended_nonzero_coordinates=int(nonzero.sum()),
          static_bf16_rounds_to_same=int(((bf==x)&nonzero).sum()),
          static_fp32_rounds_to_same=int(((fp==x)&nonzero).sum()),
          intended_abs_update_median=float(change.median()),parameter_abs_median=float(x.abs().median())))
    result=dict(status='COMPLETE_READONLY_DIAGNOSTIC',unix=time.time(),source_sha256={str(p):sha(p) for p in (A,B,C)},
      rows=rows,model_or_optimizer_mutated=False,optimizer_steps_executed=0,gpu_actions=0,
      probe_scope='float64 ideal proposal using frozen saved moments, then representation rounding; not actual past-step replay',
      inference='BF16 numerical absorption may constrain these two trainable groups; impact on navigation remains untested',
      navigation_gain_verified=False,automatic_training_allowed=False)
    with (HERE/'RESULT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    assert not torch.cuda.is_initialized();print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
