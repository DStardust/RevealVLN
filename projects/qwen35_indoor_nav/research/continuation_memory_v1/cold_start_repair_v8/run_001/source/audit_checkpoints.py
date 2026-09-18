"""CPU readback of every fixed final checkpoint and its runtime parameter updates."""
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
c=train.c


def main():
    torch.set_num_threads(4)
    run=HERE/'run_001'
    result=c.read(run/'RESULT.json')
    data=c.read(HERE.parent/'multifamily_v7/DATA.json')
    rows={}
    for key,measured in result['runs'].items():
        initial=torch.load(run/f"INITIAL_{measured['seed']}.pt",map_location='cpu',weights_only=True)
        path=run/f'{key}_MEMORY.pt'
        final=torch.load(path,map_location='cpu',weights_only=True)
        net=train.models.MemoryPolicy(2048,len(data['query_vocabulary']),8,64,.99)
        net.load_state_dict(initial)
        assert c.model_identity(net)['sha256']==measured['initial_state_sha256']
        net.load_state_dict(final)
        assert c.model_identity(net)['sha256']==measured['final_state_sha256']
        changed={}
        for name,value in final.items():
            assert bool(torch.isfinite(value).all())
            delta=value-initial[name]
            changed[name]=dict(changed_elements=int(torch.count_nonzero(delta)),
                               l2=float(delta.norm()),max_abs=float(delta.abs().max()))
        for name in ('writer.weight','recurrent.weight','action.weight'):
            assert changed[name]['changed_elements']>0,'RUNTIME_PARAMETER_NOT_UPDATED:'+key+':'+name
        rows[key]=dict(file_sha256=c.sha(path),bytes=path.stat().st_size,
                       saved_state_matches_measured=True,parameter_deltas=changed)
    c.write(HERE/'CHECKPOINT_AUDIT.json',dict(status='ALL_FIXED_CHECKPOINTS_READ_BACK',
        checkpoints=len(rows),optimizer_updates=result['optimizer_updates'],gpu_used=False,
        runtime_writer_recurrent_and_action_updated=True,runs=rows),True)
    print('Verified',len(rows),'saved checkpoints; runtime writer, recurrence and action updates confirmed.')


if __name__=='__main__':main()
