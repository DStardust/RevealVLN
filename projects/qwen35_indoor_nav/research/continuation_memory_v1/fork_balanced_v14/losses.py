"""Add a family-normalized real fork action stratum to identical V13 objectives."""
import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v14_prior_losses', HERE.parent/'cost_teacher_v13/losses.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
c, models = prior.c, prior.models
reader_logits = prior.reader_logits
state_probabilities = prior.state_probabilities
evaluate = prior.evaluate


def tensors(family, device):
    batch = prior.tensors(family, device)
    admission = prior.teacher.admit(family)
    batch['fork_cases'] = prior.teacher.branch_cases(family, admission)['cases']
    assert len(batch['fork_cases']) == 2
    return batch


def fork_loss(net, cache, batch, final):
    terms = []
    predictions = []
    for case in batch['fork_cases']:
        memory = final[[case['left_prefix'], case['right_prefix']]]
        tail = batch['family']['cells'][case['left_cell']]['tail']
        for t in range(1, case['step']+1):
            current = cache['features'][tail[t]['feature']].unsqueeze(0).expand(2, -1)
            memory, _ = net.update(current, memory)
        current = cache['features'][case['common_feature']].unsqueeze(0).expand(2, -1)
        native = cache['logits'][case['common_feature']].unsqueeze(0).expand(2, -1)
        assert int(native[0].argmax()) != 3, 'FROZEN_FORK_NATIVE_STOP_CONFLICT'
        logits = net.action_logits(memory, native, current)
        targets = torch.tensor([case['left_action'], case['right_action']], device=logits.device)
        if net.no_memory:
            assert torch.equal(logits[0], logits[1])
        terms.append(F.cross_entropy(logits, targets))
        predictions.append(int((logits.argmax(-1) == targets).all()))
    return torch.stack(terms).mean(), sum(predictions)


def losses(net, cache, batch, mode):
    original, stats, final = prior.losses(net, cache, batch, mode)
    fork, correct = fork_loss(net, cache, batch, final)
    stats.update(fork_ce=float(fork.detach()), fork_both_correct=correct,
                 fork_pairs=len(batch['fork_cases']))
    return original+fork, stats, final
