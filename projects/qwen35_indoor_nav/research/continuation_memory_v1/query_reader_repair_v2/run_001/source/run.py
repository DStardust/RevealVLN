"""One fixed reader repair on saved real causal features; no new Qwen forwards."""
import json
from pathlib import Path
import sys
import time
import traceback
import torch

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
PILOT = HERE.parent / 'pilot'
V5 = LINE / 'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.path.insert(0, str(V5))
import common as c
models = c.load('v6_reader_models', HERE / 'model.py')
training = c.load('v6_original_losses', PILOT / 'run.py')


def build(reader, data, initial, device):
    cls = models.MemoryPolicy if reader == 'mean' else models.original.MemoryPolicy
    net = cls(2048, max(x for q in data['queries'] for x in q)+1, 8, 64, .99).to(device)
    net.load_state_dict(initial)
    return net


def effective(net, memory, data, mode):
    logits, probs = [], []
    z = net.state_head(memory.flatten(1)).sigmoid()
    for cell in data['cells']:
        i = cell['prefix']
        query = torch.tensor([data['queries'][cell['query']]], device=memory.device)
        logit = net.reader(memory[i:i+1], query)[0]
        logits.append(float(logit))
        ctx = cell['query_context']
        state_prob = z[i,3] if ctx['stop_at_cutoff'] else z[i,1]
        if not ctx['stop_at_cutoff'] and ctx['suffix_anchor_before_final']:
            state_prob = torch.ones_like(state_prob)
        if not ctx['stops'] or (not ctx['stop_at_cutoff'] and not ctx['terminal_at_final']):
            state_prob = torch.zeros_like(state_prob)
        probs.append(float(state_prob if mode == 'B2' else logit.sigmoid()))
    known = [i for i,x in enumerate(data['cells']) if x['mask']]
    correct = sum((probs[i]>.5) == bool(data['cells'][i]['y']) for i in known)
    return dict(correct=correct, denominator=len(known), accuracy=correct/len(known), probabilities=probs,
                neural_reader_logits=logits)


def mechanism(net, cache, data, mode):
    """Matched legal-history swaps and causal truncations, not closed-loop claims."""
    features = torch.stack([cache['features'][p['features']] for p in data['prefixes']])
    with torch.no_grad():
        states,_ = net.encode(features)
        memory = states[:,-1]
        rows = {'correct': effective(net, memory, data, mode)}
        for length in (1,8):
            short,_ = net.encode(features[:,-length:])
            rows[f'last_{length}_observations'] = effective(net, short[:,-1], data, mode)
        wrong = memory.clone()
        lookup = {(p['history_id'],p['task_id']):i for i,p in enumerate(data['prefixes'])}
        for i,p in enumerate(data['prefixes']):
            donor = 'H_A' if p['history_id']=='H_B' else 'H_B'
            wrong[i] = memory[lookup[(donor,p['task_id'])]]
        rows['wrong_history_same_task'] = effective(net, wrong, data, mode)
        sham = memory.clone()
        eligible = []
        for i,p in enumerate(data['prefixes']):
            if p['history_id'] in ('H_A','H_A_I'):
                donor = 'H_A_I' if p['history_id']=='H_A' else 'H_A'
                sham[i] = memory[lookup[(donor,p['task_id'])]]
                eligible.extend(j for j,x in enumerate(data['cells']) if x['prefix']==i and x['mask'])
        sham_result = effective(net, sham, data, mode)
        rows['same_state_sham'] = dict(eligible_cells=eligible,
            changed_predictions=sum((sham_result['probabilities'][i]>.5)!=(rows['correct']['probabilities'][i]>.5) for i in eligible),
            probabilities=[sham_result['probabilities'][i] for i in eligible])
        base = torch.stack([cache['logits'][p['features'][-1]] for p in data['prefixes']])
        action = net.action_logits(memory,base)
        saved = memory.clone()
        for q in data['queries']:
            net.reader(memory[:1],torch.tensor([q],device=memory.device))
        assert torch.equal(saved,memory) and torch.equal(action,net.action_logits(memory,base))
        rows['action_memory_perturbation'] = dict(wrong_history_max_logit_delta=float((action-net.action_logits(wrong,base)).abs().max()),
            sham_max_logit_delta=float((action-net.action_logits(sham,base)).abs().max()),
            closed_loop_or_correct_action_test=False)
    rows['scope'] = 'Exposed single-family teacher-forced memory diagnostic; swaps do not establish spatial/task disentanglement'
    return rows


def diagnose_reader(initial, data, device):
    # Same already-trained state: isolate summary operation without any update.
    source = torch.load(PILOT/'run_002/Ours_MEMORY.pt',map_location='cpu',weights_only=True)
    result = {}
    for reader in ('last','mean'):
        net = build(reader,data,source,device)
        summaries, gradients = [], []
        for tokens in data['queries']:
            embedded = net.query_embedding(torch.tensor([tokens],device=device))
            embedded.retain_grad()
            outputs, hidden = net.query_gru(embedded)
            summary = hidden[-1] if reader=='last' else outputs.mean(1)
            summary.square().sum().backward()
            summaries.append(summary.detach())
            gradients.append(dict(tokens=len(tokens), first20_gradient_norm=float(embedded.grad[:,:20].norm()),
                last20_gradient_norm=float(embedded.grad[:,-20:].norm())))
            net.zero_grad(set_to_none=True)
        result[reader] = dict(queries=gradients,
            max_summary_delta_from_q0=[float((summaries[0]-x).abs().max()) for x in summaries[1:]])
    return result


def main(run):
    config = c.read(HERE/'CONFIG.json')
    data = c.read(PILOT/'DATA.json')
    previous = c.read(PILOT/'run_002/RESULT.json')
    for path, digest in config['source_hashes'].items():
        assert c.sha(LINE/path)==digest, 'BOUND_SOURCE_CHANGED:'+path
    assert c.sha(PILOT/'DATA.json')==previous['shared_data_sha256']
    assert c.sha(PILOT/'run_002/FEATURES.pt')==previous['shared_feature_sha256']
    torch.manual_seed(1209)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0'
    cache = {k:v.to(device) for k,v in torch.load(PILOT/'run_002/FEATURES.pt',map_location='cpu',weights_only=True).items()}
    initial = torch.load(PILOT/'run_002/INITIAL_MEMORY.pt',map_location='cpu',weights_only=True)
    c.write(run/'RUNTIME_IDENTITY.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),gpu_uuid=c.read(run/'PREFLIGHT.json')['selected']['uuid'],
        feature_sha256=previous['shared_feature_sha256'], data_sha256=previous['shared_data_sha256'],
        base_encoder_loaded=False,new_qwen_forwards=0,dtype='float32',tf32=False,deterministic_algorithms=True),True)
    c.write(run/'READER_DIAGNOSIS.json',diagnose_reader(initial,data,device),True)
    result = {}
    for arm in config['arms']:
        net = build(arm['reader'],data,initial,device)
        initial_id = c.model_identity(net)['sha256']
        assert initial_id==previous['arms']['B2']['initial_state_sha256']
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        start = time.monotonic()
        with torch.no_grad(): _,before,_ = training.losses(net,cache,data,arm['loss'])
        for step in range(100):
            net.zero_grad(set_to_none=True)
            loss,stats,_ = training.losses(net,cache,data,arm['loss'])
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
            opt.step()
            c.append(run/(arm['id']+'_STEPS.jsonl'),dict(step=step+1,loss=float(loss.detach()),**stats))
            c.write(run/'PROGRESS.json',dict(unix=time.time(),arm=arm['id'],step=step+1))
        with torch.no_grad(): _,after,_ = training.losses(net,cache,data,arm['loss'])
        state = {k:v.detach().cpu() for k,v in net.state_dict().items()}
        torch.save(state,run/(arm['id']+'_MEMORY.pt'))
        final_id = c.model_identity(net)['sha256']
        old_name = 'B2' if arm['loss']=='B2' else 'Ours'
        result[arm['id']] = dict(initial=before,final=after,updates=100,initial_state_sha256=initial_id,
            final_state_sha256=final_id,seconds=time.monotonic()-start,
            identical_to_prior_checkpoint=final_id==previous['arms'][old_name]['final_state_sha256'],
            mechanism=mechanism(net,cache,data,arm['loss']))
        c.write(run/(arm['id']+'_RESULT.json'),result[arm['id']],True)
        del net,opt,loss
        torch.cuda.empty_cache()
    c.write(run/'RESULT.json',dict(status='FIXED_READER_REPAIR_COMPARISON_COMPLETE',arms=result,
        optimizer_updates=300,new_qwen_forwards=0,new_environment_decisions=0,
        independent_test=False,closed_loop_memory_gain_tested=False,original_training_admission=False,
        scientific_superiority='UNTESTED',hyperparameter_search_trials=0),True)


if __name__ == '__main__':
    run = Path(sys.argv[1])
    try:
        main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
