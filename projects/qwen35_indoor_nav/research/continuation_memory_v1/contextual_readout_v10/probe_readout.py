"""CPU bottleneck diagnosis; copied probe heads never enter an action policy.

All runtime parameters and causal cached features stay frozen. Probe gradients
intentionally stop at memory: this measures readability, not training old writes.
"""
import copy
from pathlib import Path
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train
c = train.c


def task_results(state, batch):
    rows = []
    for i, ctx in enumerate(batch['contexts']):
        z = state[batch['cell_prefix'][i]]
        p = z[3] if ctx['stop_at_cutoff'] else z[1]
        if not ctx['stop_at_cutoff'] and ctx['suffix_anchor_before_final']:
            p = torch.ones_like(p)
        if not ctx['stops'] or (not ctx['stop_at_cutoff'] and not ctx['terminal_at_final']):
            p = torch.zeros_like(p)
        rows.append(p)
    return torch.stack(rows)


def measure(probabilities, families):
    result = {}
    for split in ('fit', 'check'):
        pairs = correct = cells = both = 0
        per_family = []
        for probability, family in zip(probabilities, families):
            if family['split'] != split:
                continue
            predicted = probability > .5
            count = right = 0
            for i, left in enumerate(family['cells']):
                if not left['mask']:
                    continue
                cells += 1
                correct += int(bool(predicted[i]) == bool(left['y']))
                prefix = family['prefixes'][left['prefix']]
                if prefix['history_id'] != 'H_A':
                    continue
                for j, other in enumerate(family['cells']):
                    donor = family['prefixes'][other['prefix']]
                    if (other['mask'] and donor['history_id'] == 'H_B'
                            and prefix['task_id'] == donor['task_id']
                            and left['query'] == other['query'] and left['y'] != other['y']):
                        assert prefix['features'][-1] == donor['features'][-1]
                        count += 1
                        right += int(bool(predicted[i]) == bool(left['y']) and bool(predicted[j]) == bool(other['y']))
            pairs += count
            both += right
            per_family.append(dict(family_id=family['family_id'], house=family['house'],
                                   opposing_pairs=count, both_correct=right))
        result[split] = dict(cells=cells, correct=correct, accuracy=correct/cells,
                             opposing_pairs=pairs, both_correct=both, families=per_family)
    return result


def fit_probe(inputs, targets, masks, families, seed, kind, config, key):
    selected = [i for i, family in enumerate(families) if family['split'] == 'fit']
    x = torch.cat([inputs[i] for i in selected])
    y = torch.cat([targets[i] for i in selected])
    mask = torch.cat([masks[i] for i in selected]).bool()
    x, y = x[mask], y[mask]
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-4)
    x = (x-mean)/scale
    torch.manual_seed(seed+10000)
    head = nn.Linear(x.shape[-1], 4) if kind == 'state' else nn.Sequential(
        nn.Linear(x.shape[-1], 128), nn.Tanh(), nn.Linear(128, 1))
    initial = c.model_identity(head)['sha256']
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    for step in range(1, config['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        if kind == 'outcome':
            logits = logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
        optimizer.step()
        if step == 1 or step % 50 == 0:
            c.append(HERE/'READOUT_PROBE_STEPS.jsonl', dict(key=key, step=step, fit_loss=float(loss.detach())))
            c.write(HERE/'READOUT_PROBE_PROGRESS.json', dict(unix=time.time(), key=key, step=step))
    final = c.model_identity(head)['sha256']
    assert initial != final
    with torch.no_grad():
        outputs = [head((value-mean)/scale).sigmoid() for value in inputs]
    return outputs, dict(initial_state_sha256=initial, final_state_sha256=final,
                         parameter_updated=True, optimizer_updates=config['steps'], fit_examples=len(x))


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = c.read(HERE/'READOUT_PROBE_PROTOCOL.json')
    assert c.sha(Path(__file__)) == config['source_sha256']
    began = time.monotonic()
    prior = HERE.parent/'multifamily_v7'
    data = c.read(prior/'DATA.json')
    cache_path = prior/'run_001/FEATURES.pt'
    assert c.sha(cache_path) == config['feature_sha256']
    cache = {k: v.float() for k, v in torch.load(cache_path, map_location='cpu', weights_only=True).items()}
    families = data['families']
    batches = [train.special.tensors(family, 'cpu') for family in families]
    recorded = c.read(HERE/'run_001/RESULT.json')
    results = {}
    for seed in config['seeds']:
        for arm in config['arms']:
            key = f'{arm}_{seed}'
            net = train.models.MemoryPolicy(2048, len(data['query_vocabulary']), 8, 64, .99,
                                           no_memory=arm == 'N0').eval()
            path = HERE/'run_001'/f'{key}_MEMORY.pt'
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            expected = recorded['runs'][key]['final_state_sha256']
            assert c.model_identity(net)['sha256'] == expected
            memories, joined, old_probability = [], [], []
            with torch.no_grad():
                for batch in batches:
                    states, _ = net.encode(cache['features'][batch['prefix_indices']])
                    final = states[:, -1]
                    memories.append(final.flatten(1))
                    old_probability.append(train.special.state_probabilities(net, final, batch) if arm == 'B2'
                                           else train.special.reader_logits(net, final, batch).sigmoid())
                    if arm == 'Ours':
                        embedded = net.query_embedding(batch['queries'])
                        packed = pack_padded_sequence(embedded, batch['query_lengths'], batch_first=True, enforce_sorted=False)
                        outputs, _ = net.query_gru(packed)
                        padded, _ = pad_packed_sequence(outputs, batch_first=True)
                        summary = padded.sum(1)/torch.tensor(batch['query_lengths']).unsqueeze(1)
                        joined.append(torch.cat([final[batch['cell_prefix']].flatten(1), summary[batch['cell_query']]], -1))
            state, metadata = fit_probe(memories, [b['state_targets'][:, -1] for b in batches],
                [b['state_masks'][:, -1] for b in batches], families, seed, 'state', config, key+'_state')
            result = dict(state_probe=measure([task_results(z, b) for z, b in zip(state, batches)], families),
                          state_probe_training=metadata, checkpoint_sha256=c.sha(path),
                          runtime_state_sha256=expected, runtime_parameters_unchanged=True,
                          original_query_reader_trained=arm in ('B2', 'Ours'))
            if arm in ('B2', 'Ours'):
                result['original_reader'] = measure(old_probability, families)
            if arm == 'Ours':
                outcome, metadata = fit_probe(joined, [b['y'] for b in batches], [b['masks'] for b in batches],
                    families, seed, 'outcome', config, key+'_outcome')
                result['outcome_probe'] = measure([x.squeeze(-1) for x in outcome], families)
                result['outcome_probe_training'] = metadata
            assert c.model_identity(net)['sha256'] == expected
            results[key] = result
            print(key, {name: {split: result[name][split]['both_correct'] for split in ('fit', 'check')}
                        for name in ('original_reader', 'state_probe', 'outcome_probe') if name in result}, flush=True)
    c.write(HERE/'READOUT_PROBE_RESULT.json', dict(status='FROZEN_REPRESENTATION_CPU_DIAGNOSIS',
        protocol_sha256=c.sha(HERE/'READOUT_PROBE_PROTOCOL.json'), results=results, seconds=time.monotonic()-began,
        diagnostic_head_optimizer_updates=(len(config['arms'])+1)*len(config['seeds'])*config['steps'],
        policy_optimizer_updates=0, new_encoder_forwards=0, gpu_used=False, new_navigation_episodes=0,
        original_training_admission=False, proof_of_old_write_learning=False,
        interpretation='Held-family readability of already frozen memories; not policy memory training, closed-loop recovery, independent generalization, or publication evidence. Exact query context is used only by the offline state-probe evaluator.'), True)


if __name__ == '__main__':
    main()
