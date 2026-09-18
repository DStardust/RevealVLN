"""V7 matched supervision plus native-policy preservation in the unsupported early prefix."""
import json
from pathlib import Path
import random
import sys
import time
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
sys.path.insert(0, str(LINE / 'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c
models = c.load('v8_memory_model', HERE.parent / 'query_reader_repair_v2/model.py')


def tensors(family, device):
    prefixes, cells = family['prefixes'], family['cells']
    tails = [(i, cell) for i, cell in enumerate(cells) if cell['tail']]
    maximum = max(len(cell['tail']) for _, cell in tails)
    tail_indices, tail_targets, tail_masks, tail_alive = [], [], [], []
    for _, cell in tails:
        n = len(cell['tail'])
        tail_indices.append([e['feature'] for e in cell['tail']] + [0] * (maximum-n))
        tail_targets.append([e['target'] for e in cell['tail']] + [0] * (maximum-n))
        tail_masks.append([e['mask'] for e in cell['tail']] + [0] * (maximum-n))
        tail_alive.append([True] * n + [False] * (maximum-n))
    queries = [torch.tensor(q, device=device, dtype=torch.long) for q in family['queries']]
    result = dict(prefix_indices=[p['features'] for p in prefixes],
        state_targets=[p['state_targets'] for p in prefixes], state_masks=[p['state_masks'] for p in prefixes],
        cell_prefix=[cell['prefix'] for cell in cells], cell_query=[cell['query'] for cell in cells],
        y=[cell['y'] for cell in cells], masks=[cell['mask'] for cell in cells],
        tail_prefix=[cell['prefix'] for _, cell in tails], tail_indices=tail_indices,
        tail_targets=tail_targets, tail_masks=tail_masks, tail_alive=tail_alive)
    result = {key: torch.tensor(value, device=device) for key, value in result.items()}
    result['queries'] = pad_sequence(queries, batch_first=True)
    result['query_lengths'] = [len(q) for q in queries]
    result['contexts'] = [cell['query_context'] for cell in cells]
    result['family'] = family
    return result


def reader_logits(net, final, batch):
    embedded = net.query_embedding(batch['queries'])
    packed = pack_padded_sequence(embedded, batch['query_lengths'], batch_first=True, enforce_sorted=False)
    outputs, _ = net.query_gru(packed)
    padded, _ = pad_packed_sequence(outputs, batch_first=True)
    lengths = torch.tensor(batch['query_lengths'], device=final.device)
    summaries = padded.sum(1) / lengths.unsqueeze(1)
    joined = torch.cat([final[batch['cell_prefix']].flatten(1), summaries[batch['cell_query']]], -1)
    return net.result_head(joined).squeeze(-1)


def state_probabilities(net, final, batch):
    state = net.state_head(final.flatten(1)).sigmoid()
    rows = []
    for i, ctx in enumerate(batch['contexts']):
        z = state[batch['cell_prefix'][i]]
        probability = z[3] if ctx['stop_at_cutoff'] else z[1]
        if not ctx['stop_at_cutoff'] and ctx['suffix_anchor_before_final']:
            probability = torch.ones_like(probability)
        if not ctx['stops'] or (not ctx['stop_at_cutoff'] and not ctx['terminal_at_final']):
            probability = torch.zeros_like(probability)
        rows.append(probability)
    return torch.stack(rows)


def cold_start_preservation(net, states, native_logits, count):
    # Native logits come from the immutable causal cache, never from labels or queries.
    states=states[:, :count]
    native=native_logits[:, :count].flatten(0,1)
    corrected=net.action_logits(states.flatten(0,1),native)
    return F.kl_div(F.log_softmax(corrected,dim=-1),F.softmax(native.detach(),dim=-1),reduction='none').sum(-1).mean()


def losses(net, cache, batch, mode):
    features = cache['features'][batch['prefix_indices']]
    states, _ = net.encode(features)
    final = states[:, -1]
    query_logits = reader_logits(net, final, batch)
    masks = batch['masks'].float()
    query_loss = (F.binary_cross_entropy_with_logits(query_logits, batch['y'].float(), reduction='none') * masks).sum() / masks.sum()
    state_logits = net.state_head(states.flatten(2))
    state_masks = batch['state_masks'].float()
    state_loss = (F.binary_cross_entropy_with_logits(state_logits, batch['state_targets'].float(), reduction='none').mean(-1) * state_masks).sum() / state_masks.sum()
    # Batch independent legal tails; all arms have exactly the same owner masks.
    memory = final[batch['tail_prefix']]
    actions = []
    for t in range(batch['tail_indices'].shape[1]):
        indices = batch['tail_indices'][:, t]
        if t:
            updated, _ = net.update(cache['features'][indices], memory)
            memory = torch.where(batch['tail_alive'][:, t, None, None], updated, memory)
        actions.append(net.action_logits(memory, cache['logits'][indices]))
    action_logits = torch.stack(actions, 1)
    action_masks = batch['tail_masks'].float()
    per_action = F.cross_entropy(action_logits.flatten(0,1), batch['tail_targets'].flatten(), reduction='none').view_as(action_masks)
    action_loss = (per_action * action_masks).sum() / action_masks.sum()
    probability = state_probabilities(net, final, batch) if mode == 'B2' else query_logits.sigmoid()
    stats = dict(action_ce=float(action_loss.detach()), action_correct=int(((action_logits.argmax(-1) == batch['tail_targets']) * action_masks).sum()),
        action_owners=int(action_masks.sum()), query_correct=int(((probability > .5) == batch['y'].bool()).mul(masks).sum()),
        query_cells=int(masks.sum()), exact_state_bce=float(state_loss.detach()), crossed_result_bce=float(query_loss.detach()))
    auxiliary = {'B1': action_loss * 0, 'B2': state_loss, 'Ours': query_loss}[mode]
    preservation=cold_start_preservation(net,states,cache['logits'][batch['prefix_indices']],156)
    stats['cold_start_preservation_kl']=float(preservation.detach())
    return action_loss + auxiliary + preservation, stats, final


@torch.no_grad()
def evaluate(net, cache, batches, mode):
    rows = []
    for batch in batches:
        _, stats, final = losses(net, cache, batch, mode)
        family = batch['family']
        base = cache['logits'][batch['tail_indices']]
        masks = batch['tail_masks'].float()
        base_ce = F.cross_entropy(base.flatten(0,1), batch['tail_targets'].flatten(), reduction='none').view_as(masks)
        probabilities = state_probabilities(net, final, batch) if mode == 'B2' else reader_logits(net, final, batch).sigmoid()
        lookup = {(p['history_id'], p['task_id']): i for i,p in enumerate(family['prefixes'])}
        wrong = final.clone()
        for i,p in enumerate(family['prefixes']):
            donor = 'H_A' if p['history_id'] == 'H_B' else 'H_B'
            wrong[i] = final[lookup[(donor,p['task_id'])]]
        wrong_prob = state_probabilities(net, wrong, batch) if mode == 'B2' else reader_logits(net, wrong, batch).sigmoid()
        rows.append(dict(family_id=family['family_id'],house=family['house'],split=family['split'],**stats,
            action_accuracy=stats['action_correct']/stats['action_owners'], query_accuracy=stats['query_correct']/stats['query_cells'],
            base_action_ce=float((base_ce*masks).sum()/masks.sum()),
            base_action_correct=int(((base.argmax(-1)==batch['tail_targets'])*masks).sum()),
            probabilities=probabilities.tolist(), targets=batch['y'].tolist(),
            wrong_history_query_correct=int((((wrong_prob>.5)==batch['y'].bool())*batch['masks']).sum()),
            B1_query_reader_is_untrained=mode=='B1'))
    summaries = {}
    for split in ('fit','check'):
        selected = [row for row in rows if row['split']==split]
        summaries[split] = dict(families=len(selected),houses=sorted({r['house'] for r in selected}),
            macro_action_ce=sum(r['action_ce'] for r in selected)/len(selected),
            macro_action_accuracy=sum(r['action_accuracy'] for r in selected)/len(selected),
            query_correct=sum(r['query_correct'] for r in selected),query_cells=sum(r['query_cells'] for r in selected),
            base_macro_action_ce=sum(r['base_action_ce'] for r in selected)/len(selected),
            base_macro_action_accuracy=sum(r['base_action_correct']/r['action_owners'] for r in selected)/len(selected))
    return dict(summaries=summaries,families=rows)


def gradient_audit(net,cache,batch):
    prefixes=batch['family']['prefixes']
    index=next(i for i,p in enumerate(prefixes) if p['history_id']=='H_B' and p['task_id']=='task_B')
    critical=next(t for t,state in enumerate(prefixes[index]['state_targets']) if state[1])
    length=len(prefixes[index]['features'])
    assert length-1-critical>8,'CRITICAL_EVENT_STILL_IN_SHORT_WINDOW'
    states,writes=net.encode(cache['features'][batch['prefix_indices']],retain_steps=[critical])
    final=states[:,-1]
    logits=reader_logits(net,final,batch)
    loss=(F.binary_cross_entropy_with_logits(logits,batch['y'].float(),reduction='none')*batch['masks']).sum()/batch['masks'].sum()
    gradient=torch.autograd.grad(loss,writes[critical],retain_graph=True)[0][index]
    action=net.action_logits(final,cache['logits'][batch['prefix_indices'][:,-1]])
    action_gradient=torch.autograd.grad(action.square().sum(),final)[0]
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm())>0
    assert float(action_gradient.norm())>0
    return dict(family_id=batch['family']['family_id'],critical_step=critical,supervision_step=length-1,
        critical_writer_gradient_norm=float(gradient.norm()),action_memory_gradient_norm=float(action_gradient.norm()),
        encoder_trained=False,full_prefix_bptt=True,per_step_memory_detach=False)


def train(run, config, cache, data):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    cache={key:value.to('cuda:0').float() for key,value in cache.items()}
    batches=[tensors(family,'cuda:0') for family in data['families']]
    fit_indices=[i for i,f in enumerate(data['families']) if f['split']=='fit']
    result={}
    for seed in config['seeds']:
        torch.manual_seed(seed)
        net=models.MemoryPolicy(cache['features'].shape[1],len(data['query_vocabulary']),8,64,.99).cuda()
        initial={key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
        initial_sha=c.model_identity(net)['sha256']
        prior=c.read(HERE.parent/'multifamily_v7/run_001'/f'SCHEDULE_{seed}.json')
        assert initial_sha==prior['initial_state_sha256'],'INITIALIZATION_CHANGED'
        torch.save(initial,run/f'INITIAL_{seed}.pt')
        c.write(run/f'GRADIENT_{seed}.json',gradient_audit(net,cache,batches[fit_indices[0]]),True)
        schedule=[];rng=random.Random(seed)
        while len(schedule)<config['steps_per_arm']:
            cycle=fit_indices.copy();rng.shuffle(cycle);schedule.extend(cycle)
        schedule=schedule[:config['steps_per_arm']]
        assert schedule==prior['family_indices'],'TRAJECTORY_EXPOSURE_CHANGED'
        c.write(run/f'SCHEDULE_{seed}.json',dict(seed=seed,family_indices=schedule,initial_state_sha256=initial_sha),True)
        for arm in config['arms']:
            net.load_state_dict(initial)
            assert c.model_identity(net)['sha256']==initial_sha
            opt=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            started=time.monotonic()
            for step,index in enumerate(schedule,1):
                net.zero_grad(set_to_none=True)
                loss,stats,_=losses(net,cache,batches[index],arm)
                assert bool(torch.isfinite(loss)), 'NONFINITE_LOSS'
                loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                opt.step()
                c.append(run/f'{arm}_{seed}_STEPS.jsonl',dict(step=step,family_index=index,loss=float(loss.detach()),**stats))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='TRAIN',seed=seed,arm=arm,step=step,total=len(schedule)))
            # The check partition is evaluated only at the fixed final update.
            measured=evaluate(net,cache,batches,arm)
            final_sha=c.model_identity(net)['sha256']
            assert final_sha!=initial_sha
            torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},run/f'{arm}_{seed}_MEMORY.pt')
            measured.update(seed=seed,arm=arm,updates=len(schedule),initial_state_sha256=initial_sha,
                            final_state_sha256=final_sha,seconds=time.monotonic()-started)
            c.write(run/f'{arm}_{seed}_RESULT.json',measured,True)
            result[f'{arm}_{seed}']=measured
            del opt,loss
            torch.cuda.empty_cache()
        del net
    c.write(run/'RESULT.json',dict(status='MULTIFAMILY_MATCHED_COMPARISON_COMPLETE',runs=result,
        base_encoder_updates=0,optimizer_updates=len(config['seeds'])*len(config['arms'])*config['steps_per_arm'],
        independent_blind_test=False,closed_loop_memory_gain_tested=False,original_training_admission=False,
        selected_by_final_check_results=False),True)
