"""CPU-only matched checkpoint change audit; no model, optimizer step or GPU."""
import json
import os
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
def main():
    initial=torch.load(LINE/'sft_acceptance/v1/checkpoints/initial.pt',map_location='cpu',weights_only=True)['trainable']
    result={}
    for run in ['efficiency_run_v1','action_balance_v1']:
        # Trusted locally generated checkpoint contains explicit RNG objects.
        state=torch.load(LINE/f'sft_acceptance/{run}/rank0_terminal.pt',map_location='cpu',weights_only=False)
        assert state['updates']==200 and state['schedule_cursor']==800
        current=state['trainable'];assert current.keys()==initial.keys()
        checks=[]
        for name,value in current.items():
            previous=initial[name];assert value.shape==previous.shape and value.dtype==previous.dtype
            assert torch.isfinite(value).all()
            delta=value.float()-previous.float()
            checks.append(dict(name=name,dtype=str(value.dtype),elements=value.numel(),
                exactly_unchanged_fraction=float((value==previous).float().mean()),
                initial_norm=float(previous.float().norm()),final_norm=float(value.float().norm()),
                delta_norm=float(delta.norm()),max_abs_delta=float(delta.abs().max())))
        result[run]=dict(updates=state['updates'],parameters=checks,
            optimizer_state_dtypes=sorted({str(t.dtype) for s in state['optimizer']['state'].values() for t in s.values() if torch.is_tensor(t)}))
        del state,current
    result['scope']='posthoc CPU parameter movement only; does not establish why learning failed or precision causality'
    with (OUT/'CHECKPOINT_MOVEMENT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    for run in ['efficiency_run_v1','action_balance_v1']:
        print(json.dumps(dict(run=run,parameters=[x for x in result[run]['parameters'] if not x['name'].startswith('base.')]),allow_nan=False))
if __name__=='__main__':main()
