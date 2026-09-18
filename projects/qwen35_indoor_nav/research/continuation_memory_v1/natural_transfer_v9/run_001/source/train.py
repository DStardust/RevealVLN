"""Same V8 architecture/objectives, plus matched ordinary navigation BC."""
from pathlib import Path
import random
import sys
import time
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
sys.path.insert(0, str(LINE / 'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c
prior = c.load('v9_v8_training', HERE.parent / 'cold_start_repair_v8/train.py')
models = prior.models


def ordinary_batch(records, device):
    length = max(len(row['features']) for row in records)
    return dict(indices=torch.tensor([r['features'] + [0] * (length-len(r['features'])) for r in records], device=device),
        targets=torch.tensor([r['targets'] + [0] * (length-len(r['targets'])) for r in records], device=device),
        mask=torch.tensor([[True]*len(r['features']) + [False]*(length-len(r['features'])) for r in records], device=device))


def ordinary_loss(net, cache, batch):
    states, _ = net.encode(cache['features'][batch['indices']])
    logits = net.action_logits(states.flatten(0, 1), cache['logits'][batch['indices']].flatten(0, 1))
    per_step = F.cross_entropy(logits, batch['targets'].flatten(), reduction='none').view_as(batch['mask'])
    per_route = (per_step * batch['mask']).sum(-1) / batch['mask'].sum(-1)
    return per_route.mean(), logits.view(*batch['indices'].shape, 4)


@torch.no_grad()
def ordinary_evaluate(net, cache, records):
    rows = []
    for start in range(0, len(records), 8):
        selected = records[start:start+8]
        batch = ordinary_batch(selected, cache['features'].device)
        _, logits = ordinary_loss(net, cache, batch)
        for i, row in enumerate(selected):
            n = len(row['features'])
            target = batch['targets'][i, :n]
            base = cache['logits'][batch['indices'][i, :n]]
            current = logits[i, :n]
            rows.append(dict(record_id=row['row']['record_id'], house=row['row']['scene_group'],
                partition=row['partition'], decisions=n,
                ce=float(F.cross_entropy(current, target)), base_ce=float(F.cross_entropy(base, target)),
                correct=int((current.argmax(-1)==target).sum()), base_correct=int((base.argmax(-1)==target).sum()),
                native_action_changes=int((current.argmax(-1)!=base.argmax(-1)).sum()),
                terminal_stop_correct=bool(current[-1].argmax()==3), base_terminal_stop_correct=bool(base[-1].argmax()==3),
                premature_stops=int((current[:-1].argmax(-1)==3).sum()), base_premature_stops=int((base[:-1].argmax(-1)==3).sum())))
    summaries = {}
    for partition in ('fit', 'check'):
        selected = [r for r in rows if r['partition']==partition]
        decisions = sum(r['decisions'] for r in selected)
        summaries[partition] = dict(routes=len(selected), decisions=decisions,
            houses=sorted({r['house'] for r in selected}),
            macro_ce=sum(r['ce'] for r in selected)/len(selected),
            base_macro_ce=sum(r['base_ce'] for r in selected)/len(selected),
            accuracy=sum(r['correct'] for r in selected)/decisions,
            base_accuracy=sum(r['base_correct'] for r in selected)/decisions,
            terminal_stop_correct=sum(r['terminal_stop_correct'] for r in selected),
            base_terminal_stop_correct=sum(r['base_terminal_stop_correct'] for r in selected),
            premature_stops=sum(r['premature_stops'] for r in selected),
            base_premature_stops=sum(r['base_premature_stops'] for r in selected))
    return dict(rows=rows, summaries=summaries, scope='Teacher trajectories, not closed-loop success; current actions/labels are absent from feature input')


def train(run, config, special_cache, special_data, natural_cache, natural_data):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    special_cache = {k:v.cuda().float() for k,v in special_cache.items()}
    natural_cache = {k:v.cuda().float() for k,v in natural_cache.items()}
    batches = [prior.tensors(f, 'cuda:0') for f in special_data['families']]
    natural_fit = [i for i,r in enumerate(natural_data['records']) if r['partition']=='fit']
    result = {}
    for seed in config['seeds']:
        net = models.MemoryPolicy(2048, len(special_data['query_vocabulary']), 8, 64, .99).cuda()
        initial_path = HERE.parent / 'multifamily_v7/run_001' / f'INITIAL_{seed}.pt'
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        net.load_state_dict(initial)
        initial_sha = c.model_identity(net)['sha256']
        old_schedule = c.read(HERE.parent/'multifamily_v7/run_001'/f'SCHEDULE_{seed}.json')
        assert initial_sha == old_schedule['initial_state_sha256']
        schedule = old_schedule['family_indices']
        assert len(schedule) == config['steps_per_arm']
        rng = random.Random(seed)
        natural_order = []
        while len(natural_order) < len(schedule)*8:
            cycle = natural_fit.copy(); rng.shuffle(cycle); natural_order.extend(cycle)
        ordinary_indices = [natural_order[i*8:(i+1)*8] for i in range(len(schedule))]
        c.write(run/f'SCHEDULE_{seed}.json',dict(family_indices=schedule, ordinary_indices=ordinary_indices,
            initial_file_sha256=c.sha(initial_path), initial_state_sha256=initial_sha),True)
        c.write(run/f'GRADIENT_{seed}.json',prior.gradient_audit(net,special_cache,batches[schedule[0]]),True)
        for arm in config['arms']:
            net.load_state_dict(initial)
            assert c.model_identity(net)['sha256'] == initial_sha
            optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            began = time.monotonic()
            for step, (family_index, ordinary_indices_step) in enumerate(zip(schedule, ordinary_indices), 1):
                net.zero_grad(set_to_none=True)
                special_loss, stats, _ = prior.losses(net, special_cache, batches[family_index], arm)
                batch = ordinary_batch([natural_data['records'][i] for i in ordinary_indices_step], 'cuda:0')
                natural_loss, _ = ordinary_loss(net, natural_cache, batch)
                loss = special_loss + natural_loss
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                optimizer.step()
                c.append(run/f'{arm}_{seed}_STEPS.jsonl',dict(step=step, family_index=family_index,
                    ordinary_indices=ordinary_indices_step, loss=float(loss.detach()),ordinary_ce=float(natural_loss.detach()),**stats))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='TRAIN',seed=seed,arm=arm,step=step,total=len(schedule)))
            measured = prior.evaluate(net,special_cache,batches,arm)
            measured['ordinary'] = ordinary_evaluate(net,natural_cache,natural_data['records'])
            final_sha = c.model_identity(net)['sha256']
            assert final_sha != initial_sha
            torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},run/f'{arm}_{seed}_MEMORY.pt')
            measured.update(seed=seed,arm=arm,updates=len(schedule),initial_state_sha256=initial_sha,
                final_state_sha256=final_sha,seconds=time.monotonic()-began)
            c.write(run/f'{arm}_{seed}_RESULT.json',measured,True)
            result[f'{arm}_{seed}'] = measured
            del optimizer, loss, special_loss, natural_loss, batch
            torch.cuda.empty_cache()
        del net
    c.write(run/'RESULT.json',dict(status='MULTIFAMILY_MATCHED_COMPARISON_COMPLETE',runs=result,
        base_encoder_updates=0,optimizer_updates=len(config['seeds'])*len(config['arms'])*config['steps_per_arm'],
        original_special_training_admission=False,ordinary_labels_unchanged=True,
        selected_by_final_check_results=False,independent_blind_test=False,closed_loop_memory_gain_tested=False),True)
