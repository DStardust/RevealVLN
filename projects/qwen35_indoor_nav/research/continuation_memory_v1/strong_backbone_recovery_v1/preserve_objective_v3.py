"""Nonempty ordinary preservation and dense continuation supervision."""
import torch
import torch.nn.functional as F
from train_memory_v2 import objective as sparse_objective


def sequence_logits(model,row):
    x=row['memory_features'];memory=model.reset();queries=row['query_steps'].tolist();lookup={t:i for i,t in enumerate(queries)}
    deltas=[]
    for step in range(x.shape[0]):
        memory=model.update(x[step:step+1],memory)
        if step in lookup:
            q=lookup[step];deltas.append(model.action_delta(row['actor_features'][q:q+1],memory)[0])
    if len(deltas)!=len(queries) or not deltas:raise ValueError('EMPTY_OR_MISALIGNED_PRESERVATION')
    return row['base_logits']+torch.stack(deltas)


def preservation_loss(logits,base,target):
    if logits.shape[0]==0:raise ValueError('EMPTY_PRESERVATION_MASK')
    kl=F.kl_div(F.log_softmax(logits,-1),F.softmax(base,-1),reduction='batchmean')
    selected=logits.gather(1,target[:,None]).squeeze(1)
    alternatives=logits.masked_fill(F.one_hot(target,4).bool(),float('-inf')).max(1).values
    original_selected=base.gather(1,target[:,None]).squeeze(1)
    original_other=base.masked_fill(F.one_hot(target,4).bool(),float('-inf')).max(1).values
    desired=(original_selected-original_other).clamp(min=0,max=.25)
    margin=F.relu(alternatives-selected+desired).mean()
    return kl+margin,kl,margin


def objective(model,group,ordinary,dense,arm,weight,class_weights):
    special,sparse=sparse_objective(model,group,arm)
    regular_logits=sequence_logits(model,ordinary)
    keep,kl,margin=preservation_loss(regular_logits,ordinary['base_logits'],ordinary['targets'])
    ordinary_ce=F.cross_entropy(regular_logits,ordinary['targets'])
    dense_logits=sequence_logits(model,dense)
    dense_ce=F.cross_entropy(dense_logits,dense['targets'],weight=class_weights)
    total=special+weight*keep+.5*ordinary_ce+.5*dense_ce
    return total,dict(**sparse,ordinary_preservation_kl=kl,ordinary_margin_loss=margin,ordinary_ce=ordinary_ce,
        dense_action_ce=dense_ce,ordinary_queries=len(regular_logits),dense_queries=len(dense_logits),
        ordinary_argmax_agreement=(regular_logits.argmax(-1)==ordinary['targets']).float().mean())

