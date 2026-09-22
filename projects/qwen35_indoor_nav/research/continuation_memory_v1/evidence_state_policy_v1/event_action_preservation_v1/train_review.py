"""Verify actual event-only changes against each frozen trained ORIGINAL reference."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *


def main(run):
    cfg=config(run);records=[]
    for seed in cfg['seeds']:
        source=Path(cfg['source_original_run'])/'train'/f'ORIGINAL_{seed}'
        source_result=read(source/'RESULT.json')
        source_state=torch.load(source/'FINAL.pt',map_location='cpu',weights_only=True)
        for arm in cfg['arms']:
            tag=f'{arm}_{seed}';folder=run/'train'/tag;result=read(folder/'RESULT.json')
            net=make_head(tag)
            initial=torch.load(folder/'INITIAL.pt',map_location='cpu',weights_only=True)
            final=torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True)
            if any(not torch.equal(initial[k],v) for k,v in source_state.items()):raise ValueError('WRONG_TRAINED_ORIGINAL_INITIALIZATION')
            net.load_state_dict(final)
            if result['updates']!=cfg['model_updates'][arm] or c.model_identity(net)['sha256']!=result['final']:
                raise ValueError('MODEL_UPDATE_OR_STATE_MISMATCH')
            if sha(folder/'FINAL.pt')!=result['checkpoint_sha256'] or result['initial']!=source_result['final']:
                raise ValueError('CHECKPOINT_OR_INITIAL_IDENTITY')
            changed=[k for k in initial if not torch.equal(initial[k],final[k])]
            if arm=='ORIGINAL':
                if changed or result['updates']!=0:raise ValueError('FROZEN_REFERENCE_CHANGED')
            else:
                if not changed or any(not k.startswith('events.') for k in changed):raise ValueError('UPDATE_ESCAPED_EVENT_MODULE')
                if not result['frozen_unchanged'] or result['frozen_before']!=result['frozen_after']:raise ValueError('FROZEN_STATE_CHANGED')
                expected={n for n,p in net.named_parameters() if n.startswith('events.')}
                if set(result['trainable_parameters'])!=expected:raise ValueError('OPTIMIZER_SCOPE_CHANGED')
                checkpoints=[p for p in folder.glob('attempt_*/STEP_*.json')]
                if not set(range(0,cfg['steps']+1,cfg['checkpoint_every']))<={read(p)['step'] for p in checkpoints}:
                    raise ValueError('MISSING_RESUMABLE_CHECKPOINT')
                for p in checkpoints:
                    if sha(p.with_suffix('.pt'))!=read(p)['sha256']:raise ValueError('CHECKPOINT_SEAL')
            records.append(dict(model=tag,updates=result['updates'],changed_parameters=changed,source_original=source_result['final'],
                                final=result['final'],frozen_non_event_unchanged=True))
    immutable(run/'TRAINING_REVIEW.json',dict(status='EVENT_ONLY_REPAIR_TRAINING_VERIFIED',models=records,
        new_updates=sum(r['updates'] for r in records),base_updates=0,source_original_updates_per_seed=1200,
        initialization_matched=True,dev_used_for_selection=False,
        note='Only eventMLP weights changed. Frozen weights do not imply unchanged predicted states/actions; complete closed-loop pairing is required.'))
    print('Three event-only updates verified; three reference heads unchanged.',flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
