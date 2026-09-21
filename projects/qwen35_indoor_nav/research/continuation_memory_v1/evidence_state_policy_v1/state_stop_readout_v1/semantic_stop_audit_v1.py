"""Read-only correction: teacher disagreement is not necessarily premature STOP."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def metrics(actions,targets,ready):
    return dict(n=len(actions),ready_n=int(ready.sum()),not_ready_n=int((~ready).sum()),
        teacher_stop_n=int((targets==3).sum()),teacher_continue_n=int((targets!=3).sum()),
        teacher_missed_stop=int(((targets==3)&(actions!=3)).sum()),teacher_stop_disagreement=int(((targets!=3)&(actions==3)).sum()),
        premature_stop=int(((~ready)&(actions==3)).sum()),ready_defer=int((ready&(actions!=3)).sum()),
        ready_stop=int((ready&(actions==3)).sum()),ready_teacher_continue=int((ready&(targets!=3)).sum()))

def main(run):
    import torch
    torch.set_num_threads(2)
    module=load('semantic_audit_readout',HERE/'model.py');training=load('semantic_audit_train',HERE/'train.py')
    binding=read(run/'BINDING.json')
    if sha(run/'DATA.json')!=binding['files'][str(run/'DATA.json')]:raise ValueError('DATA_CHANGED')
    records=[]
    for seed in read(run/'PROTOCOL.json')['seeds']:
        path=run/'cache'/f'STATE_{seed}.pt'
        if sha(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('CACHE_CHANGED')
        cache=torch.load(path,map_location='cpu',weights_only=True)
        final=run/'train'/f'STOPFIX_{seed}'/'FINAL.pt'
        if sha(final)!=read(final.parent/'RESULT.json')['checkpoint_sha256']:raise ValueError('HEAD_CHANGED')
        head=module.StopReadout(seed);weights=torch.load(final,map_location='cpu',weights_only=True)
        head.load_state_dict({k.removeprefix('readout.'):v for k,v in weights.items() if k.startswith('readout.')})
        for split in ('FIT','DEV'):
            for task in ('task_A','task_T'):
                rows=[r for f in cache['families'].values() if f['split']==split for r in f['rows'] if r['task']==task and bool(r['action_mask'].any())]
                inputs=training.stack(rows,'cpu',True);truth=torch.cat([r['truth'][r['action_mask']] for r in rows]);ready=truth[:,3].bool()
                with torch.inference_mode():actions=head(inputs['logits'],inputs['state']).argmax(-1)
                records.append(dict(seed=seed,split=split,task=task,original=metrics(inputs['logits'].argmax(-1),inputs['targets'],ready),repair=metrics(actions,inputs['targets'],ready)))
    result=dict(status='READ_ONLY_SEMANTIC_STOP_AUDIT',source_evaluator=str(V16/'evaluator_v16.py'),source_evaluator_sha256=sha(V16/'evaluator_v16.py'),
        rule='ready is exact registered state_sequence bit 4. safe endpoint also needs structure, collision-free trajectory and actual STOP within 500. No synthetic STOP is recorded as a physical rollout.',
        new_updates=0,new_gpu_hours=0,records=records,
        correction='Earlier false_stop was teacher STOP disagreement, not task-semantic premature STOP. Earlier local gate can reject legal early stops. Original result preserved; no method gain or replacement labels inferred.')
    immutable(run/'SEMANTIC_STOP_AUDIT.json',result)
    print(result)

if __name__=='__main__':main(Path(sys.argv[1]))
