"""Current SEE2 observability probe on immutable causal features; never a policy."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
sys.path.insert(0, str(HERE))
import train
c = train.c


def examples(data):
    """Recover current two-frame witnesses, not changes in cumulative state."""
    loader_module = c.load('witness_loader', LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
    core = c.load('witness_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')
    rows, known_labels = [], {}
    excluded_start = unknown = 0
    for family, audit in zip(data['families'], data['audit']['families']):
        root = LINE/audit['export']
        assert c.sha(root/'MANIFEST.json') == audit['manifest_sha256']
        manifest = c.read(root/'MANIFEST.json')
        assert manifest['family_id'] == family['family_id']
        compiler = core.Compiler(**manifest['compiler_config'])
        loader = loader_module.FamilyLoader(root, compiler)
        unique = {}
        for prefix in family['prefixes']:
            cell = next(x for x in loader.cells if x['history_id'] == prefix['history_id']
                        and x['task_id'] == prefix['task_id'])
            trace = loader.read(cell['trace_path'])
            cutoff = cell['prefix_cutoff']
            events = compiler.atoms(trace['observations'][:cutoff+1])
            records = loader.prefix_records(cell['prefix_id'])
            assert len(records) == len(events) == len(prefix['features'])
            task = compiler.tasks[prefix['task_id']]
            for t, (record, feature, event) in enumerate(zip(records, prefix['features'], events)):
                clean = dict(instruction=record['instruction'],
                    rgb_refs=[o['rgb_ref'] for o in record['observations']],
                    executed=[a['action'].lower() for a in record['executed_actions']])
                key = hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()
                assert data['features'][feature]['key'] == key, 'FEATURE_ALIGNMENT'
                if t == 0:
                    excluded_start += 1
                    continue
                labels = [event[task['anchor']], event[task['terminal']]]
                mask = [x is not None for x in labels]
                unknown += sum(not x for x in mask)
                y = [int(bool(x)) for x in labels]
                for j, valid in enumerate(mask):
                    if valid:
                        assert known_labels.setdefault((key, j), y[j]) == y[j], 'SAME_INPUT_CONFLICT'
                if feature in unique:
                    assert unique[feature]['y'] == y and unique[feature]['mask'] == mask
                else:
                    unique[feature] = dict(family_id=family['family_id'], house=family['house'],
                        split=family['split'], feature=feature, y=y, mask=mask)
        rows.extend(unique.values())
        print('LABELS', len(rows), family['family_id'], flush=True)
    assert not ({r['feature'] for r in rows if r['split']=='fit'} &
                {r['feature'] for r in rows if r['split']=='check'}), 'CROSS_SPLIT_INPUT_OVERLAP'
    return rows, dict(excluded_single_frame_records=excluded_start, unknown_labels_masked=unknown,
                     deduplicated_within_family=True, same_input_labels_consistent=True)


def confusion(logits, labels, masks, index):
    result = {}
    for j, name in enumerate(('anchor_now', 'terminal_now')):
        selected = index & masks[:, j]
        pred, truth = logits[selected, j] > 0, labels[selected, j].bool()
        tp, tn = int((pred & truth).sum()), int((~pred & ~truth).sum())
        fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
        positive, negative = tp+fn, tn+fp
        recall = tp/positive if positive else None
        specificity = tn/negative if negative else None
        result[name] = dict(tp=tp, tn=tn, fp=fp, fn=fn, positives=positive, negatives=negative,
            recall=recall, specificity=specificity,
            accuracy=(tp+tn)/(positive+negative) if positive+negative else None,
            balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = c.read(HERE/'WITNESS_PROBE_PROTOCOL.json')
    for relative, digest in config['sources'].items():
        assert c.sha(LINE/relative) == digest, relative
    run = HERE/'witness_probe_run_001'
    run.mkdir(exist_ok=False)
    began = time.monotonic()
    data = c.read(HERE.parent/'multifamily_v7/DATA.json')
    rows, audit = examples(data)
    c.write(run/'EXAMPLES.json', dict(rows=rows, audit=audit), True)
    cache = torch.load(HERE.parent/'multifamily_v7/run_001/FEATURES.pt', map_location='cpu', weights_only=True)
    x = cache['features'][[r['feature'] for r in rows]].float()
    y = torch.tensor([r['y'] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r['mask'] for r in rows], dtype=torch.bool)
    fit = torch.tensor([r['split']=='fit' for r in rows])
    mean, scale = x[fit].mean(0), x[fit].std(0, unbiased=False).clamp_min(1e-4)
    x = (x-mean)/scale
    weight = torch.zeros_like(y)
    # Equal contribution per FIT family and available label class for each target.
    for family in data['families']:
        if family['split'] != 'fit':
            continue
        family_mask = torch.tensor([r['family_id']==family['family_id'] for r in rows])
        for j in range(2):
            for label in (0, 1):
                index = family_mask & mask[:, j] & (y[:, j]==label)
                if index.any():
                    weight[index, j] = 1/int(index.sum())
    weight *= int(fit.sum()) / weight.sum(0)
    fit_indices = fit.nonzero().flatten()
    results = {}
    for seed in config['seeds']:
        generator = torch.Generator().manual_seed(seed)
        schedule = torch.randint(len(fit_indices), (config['steps'], config['batch_size']), generator=generator)
        for kind in config['heads']:
            torch.manual_seed(seed)
            head = nn.Linear(2048, 2) if kind == 'linear' else nn.Sequential(
                nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 2))
            initial = c.model_identity(head)['sha256']
            optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            for step, chosen in enumerate(schedule, 1):
                selected = fit_indices[chosen]
                optimizer.zero_grad(set_to_none=True)
                loss = (F.binary_cross_entropy_with_logits(head(x[selected]), y[selected], reduction='none')
                        * weight[selected]).mean()
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    c.append(run/'STEPS.jsonl', dict(seed=seed, head=kind, step=step, loss=float(loss.detach())))
                    c.write(run/'PROGRESS.json', dict(unix=time.time(), seed=seed, head=kind, step=step))
            final = c.model_identity(head)['sha256']
            assert initial != final
            with torch.no_grad():
                logits = torch.cat([head(chunk) for chunk in x.split(512)])
            key = f'{kind}_{seed}'
            torch.save(dict(head=head.state_dict(), mean=mean, scale=scale), run/f'{key}.pt')
            c.write(run/f'{key}_PREDICTIONS.json', logits.tolist(), True)
            result = dict(initial_state_sha256=initial, final_state_sha256=final,
                          optimizer_updates=config['steps'], parameters_updated=True)
            result['splits'] = {name: confusion(logits, y, mask, fit if name=='fit' else ~fit)
                                for name in ('fit', 'check')}
            result['houses'] = {house: confusion(logits, y, mask,
                torch.tensor([r['house']==house for r in rows])) for house in sorted({r['house'] for r in rows})}
            result['families'] = {fid: confusion(logits, y, mask,
                torch.tensor([r['family_id']==fid for r in rows])) for fid in sorted({r['family_id'] for r in rows})}
            results[key] = result
            print(key, result['splits'], flush=True)
            c.write(run/f'{key}_RESULT.json', result, True)
    c.write(run/'RESULT.json', dict(status='CURRENT_WITNESS_CPU_PROBE_COMPLETE',
        protocol_sha256=c.sha(HERE/'WITNESS_PROBE_PROTOCOL.json'), results=results,
        seconds=time.monotonic()-began, diagnostic_head_updates=config['steps']*len(results),
        policy_updates=0, new_qwen_forwards=0, new_navigation_episodes=0, gpu_hours=0,
        original_training_admission=False, scientific_method_benefit=False,
        interpretation='Diagnostic readability of current two-frame witnesses on exposed held-house families; not long-memory learning, action use, closed-loop improvement, or paper validation.'), True)


if __name__ == '__main__':
    main()
