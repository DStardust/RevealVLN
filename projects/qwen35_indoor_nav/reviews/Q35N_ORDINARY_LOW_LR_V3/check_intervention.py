"""CPU-only same-200-update parameter drift check; not a navigation outcome."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(2)
    paths={
        'start':LINE/'sft_acceptance/ordinary_expanded_v1/formal/attempt_001/checkpoint_000004000.pt',
        'high':LINE/'sft_acceptance/ordinary_expanded_continue_v2/formal/attempt_001/checkpoint_000004200.pt',
        'low':LINE/'sft_acceptance/ordinary_expanded_low_lr_v3/formal/attempt_001/checkpoint_000004200.pt'}
    states={};hashes={}
    for k,p in paths.items():
        hashes[k]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashes[k]==json.loads(Path(str(p)+'.json').read_text())['sha256']
        states[k]=torch.load(p,map_location='cpu',weights_only=True)
    assert states['high']['cursor']==states['low']['cursor']
    assert states['high']['global_decisions']==states['low']['global_decisions']==358092
    rows=[];summary=collections.defaultdict(lambda:dict(base_sq=0.,high_sq=0.,low_sq=0.,tensors=0))
    dtypes=collections.Counter()
    for key,base in states['start']['trainable'].items():
        a=states['high']['trainable'][key].double()-base.double()
        b=states['low']['trainable'][key].double()-base.double()
        group='lora_A' if 'lora_A' in key else 'lora_B' if 'lora_B' in key else 'action_head' if 'action_head' in key else 'exec_embed' if 'exec_embed' in key else 'action_query'
        x=summary[group];x['base_sq']+=float(base.double().square().sum());x['high_sq']+=float(a.square().sum());x['low_sq']+=float(b.square().sum());x['tensors']+=1
        dtypes[str(base.dtype)]+=base.numel()
        rows.append(dict(name=key,dtype=str(base.dtype),high_change_norm=float(a.norm()),low_change_norm=float(b.norm())))
    for x in summary.values():
        x['high_relative_l2']=(x['high_sq']/x['base_sq'])**.5
        x['low_relative_l2']=(x['low_sq']/x['base_sq'])**.5
        x['low_to_high_update_norm_ratio']=(x['low_sq']/x['high_sq'])**.5
    result=dict(status='PARAMETER_INTERVENTION_MEASURED',unix=time.time(),checkpoint_hashes=hashes,
        identical_200_update_data_exposure=True,high_lr=states['high']['optimizer']['param_groups'][0]['lr'],
        low_lr=states['low']['optimizer']['param_groups'][0]['lr'],groups=dict(summary),trainable_dtype_numel=dict(dtypes),
        parameters=rows,gpu_initialized=torch.cuda.is_initialized(),navigation_gain_verified=False,
        interpretation='Confirms reduced parameter movement, not proof of better generalization or cause of the high-LR failure')
    with (HERE/'PARAMETER_INTERVENTION_CHECK.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','high_lr','low_lr','groups','trainable_dtype_numel')},ensure_ascii=False))


if __name__=='__main__':main()
