"""Check complete matched optimization and immutable initialization, without DEV selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    import torch
    cfg=config(run);records=[]
    for seed in cfg['seeds']:
        initials=[]
        for arm in cfg['arms']:
            folder=run/'train'/f'{arm}_{seed}';r=read(folder/'RESULT.json');net=make_head(f'{arm}_{seed}')
            net.load_state_dict(torch.load(folder/'INITIAL.pt',map_location='cpu',weights_only=True))
            initial=c.model_identity(net)['sha256'];initials.append(initial)
            net.load_state_dict(torch.load(folder/'FINAL.pt',map_location='cpu',weights_only=True))
            if initial!=r['initial'] or c.model_identity(net)['sha256']!=r['final']:raise ValueError('TRAINED_STATE_IDENTITY')
            if r['updates']!=cfg['steps'] or r['base_updates']!=0 or r['base_loaded'] or not r['parameters_changed']:raise ValueError('INCOMPLETE_OR_WRONG_TRAINING')
            if sha(folder/'FINAL.pt')!=r['checkpoint_sha256']:raise ValueError('FINAL_FILE_CHANGED')
            checkpoints=[p for p in folder.glob('attempt_*/STEP_*.json')]
            steps={read(p)['step'] for p in checkpoints}
            if not set(range(200,1201,200))<=steps:raise ValueError('MISSING_RECOVERABLE_CHECKPOINT')
            for p in checkpoints:
                if sha(p.with_suffix('.pt'))!=read(p)['sha256']:raise ValueError('CHECKPOINT_SEAL')
            records.append(dict(arm=arm,seed=seed,**r))
        if len(set(initials))!=1:raise ValueError('PAIRED_INITIALIZATION_CHANGED')
    a=read(run/'train'/f'ORIGINAL_{cfg["seeds"][0]}'/'FIT_WEIGHTS.json')
    for r in records:
        if read(run/'train'/f'{r["arm"]}_{r["seed"]}'/'FIT_WEIGHTS.json')!=a:raise ValueError('AUXILIARY_WEIGHTS_CHANGED')
    immutable(run/'TRAINING_REVIEW.json',dict(status='MATCHED_TRAINING_COMPLETE',models=records,updates=sum(r['updates'] for r in records),
        base_updates=0,initialization_matched=True,common_auxiliary_weights=True,dev_used_for_selection=False,
        validation='Parameter updates and complete checkpoints; no efficacy conclusion until full registered DEV.'))

if __name__=='__main__':main(Path(sys.argv[1]))
