"""V11 architecture and losses; identical minimum-cost actor admission for all arms."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import teacher
spec=importlib.util.spec_from_file_location('v13_prior_losses',HERE.parent/'query_semantics_v11/losses.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
c=prior.c
models=prior.models
losses=prior.losses
evaluate=prior.evaluate
reader_logits=prior.reader_logits
state_probabilities=prior.state_probabilities


def tensors(family,device):
    batch=prior.tensors(family,device)
    admission=teacher.admit(family)
    tails=[i for i,cell in enumerate(family['cells']) if cell['tail']]
    batch['tail_masks'].zero_()
    for row,index in enumerate(tails):
        mask=admission['masks'][index]
        batch['tail_masks'][row,:len(mask)]=batch['tail_masks'].new_tensor(mask)
    assert int(batch['tail_masks'].sum())==admission['admitted_actions']
    return batch
