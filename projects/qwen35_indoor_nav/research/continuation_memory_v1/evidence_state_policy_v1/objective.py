"""Identical action, event and exact-state supervision for all new arms."""
import torch
from torch.nn import functional as F
from common import old


def batch(family, device):
    b = old.batch(family, device)
    maximum = b['alive'].shape[1]
    for key, target in (('event_targets', 'events'), ('event_masks', 'event_mask')):
        b[target] = torch.tensor([r[key]+[[0, 0]]*(maximum-len(r[key])) for r in family['sequences']], dtype=torch.float32, device=device)
    b['preservation_mask'] *= b['alive']
    return b


def weights(families):
    result = old.class_weights(families)
    positive = [0, 0]; negative = [0, 0]
    for f in families:
        if f['split'] != 'FIT':
            raise ValueError('NONFIT_LOSS_STATISTICS')
        for r in f['sequences']:
            for y, m in zip(r['event_targets'], r['event_masks']):
                for k in range(2):
                    positive[k] += y[k]*m[k]; negative[k] += (1-y[k])*m[k]
    if min(positive+negative) <= 0:
        raise ValueError('MISSING_FIT_EVENT_CLASS')
    result['event'] = [[(p+n)/(2*n), (p+n)/(2*p)] for p,n in zip(positive, negative)]
    return result


def losses(net, cache, b, w):
    result = net(cache['features'][b['indices']], cache['logits'][b['indices']], b['alive'])
    logits = result['logits']
    ce = F.cross_entropy(logits.flatten(0, 1), b['targets'].flatten(), reduction='none').view_as(b['targets'])
    action = old.average(ce, b['action_mask'])
    cut = torch.zeros_like(b['action_mask'])
    for i, row in enumerate(b['family']['sequences']):
        cut[i, row['cutoff']] = b['action_mask'][i, row['cutoff']]
    cutoff = old.average(ce, cut)
    native = cache['logits'][b['indices']]
    kl = old.average(F.kl_div(logits.log_softmax(-1), native.softmax(-1), reduction='none').sum(-1), b['preservation_mask'])
    sw = torch.tensor(w['state'], device=logits.device)
    sm = torch.where(b['state'].bool(), sw[:, 1], sw[:, 0]) * b['state_mask'][..., None]
    state = old.average(F.binary_cross_entropy(result['state'].clamp(1e-6, 1-1e-6), b['state'], reduction='none'), sm)
    ew = torch.tensor(w['event'], device=logits.device)
    em = torch.where(b['events'].bool(), ew[:, 1], ew[:, 0]) * b['event_mask']
    event = old.average(F.binary_cross_entropy_with_logits(result['event_logits'], b['events'], reduction='none'), em)
    loss = action + cutoff + kl + state + event
    result.update(action_loss=action+cutoff, state_loss=state, event_loss=event)
    stats = dict(action_ce=float(action.detach()), cutoff_ce=float(cutoff.detach()), preservation_kl=float(kl.detach()),
                 state_bce=float(state.detach()), event_bce=float(event.detach()), action_count=int(b['action_mask'].sum()),
                 event_labels=int(b['event_mask'].sum()), state_labels=int(b['state_mask'].sum())*4)
    return loss, stats, result


def ordinary_loss(net, cache, records):
    maximum = max(len(r['features']) for r in records); device = cache['features'].device
    indices = torch.tensor([r['features']+[0]*(maximum-len(r['features'])) for r in records], device=device)
    targets = torch.tensor([r['targets']+[0]*(maximum-len(r['targets'])) for r in records], device=device)
    alive = torch.tensor([[True]*len(r['features'])+[False]*(maximum-len(r['features'])) for r in records], device=device)
    logits = net(cache['features'][indices], cache['logits'][indices], alive)['logits']
    ce = F.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='none').view_as(alive)
    return ((ce*alive).sum(-1)/alive.sum(-1)).mean()
