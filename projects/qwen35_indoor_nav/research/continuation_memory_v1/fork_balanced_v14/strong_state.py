"""Capacity-matched nonlinear exact-state falsifier; frozen V14 actors stay intact."""
import os
from pathlib import Path
import sys
import time
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train
c = train.c


def initialize(seed, data):
    torch.manual_seed(seed)
    net = train.models.MemoryPolicy(2048, len(data['query_vocabulary']), 8, 64, .99)
    shared = torch.load(HERE.parent/'query_semantics_v11/run_001'/f'INITIAL_{seed}.pt',
                        map_location='cpu', weights_only=True)
    net.load_state_dict(shared)
    net.state_head = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 4))
    with torch.no_grad():
        # Same initial memory-coordinate mapping as the outcome MLP; the state
        # head has no future-query columns. Four exact bits retain their labels.
        net.state_head[0].weight.copy_(shared['result_head.0.weight'][:, :512])
        net.state_head[0].bias.copy_(shared['result_head.0.bias'])
    for name, value in net.state_dict().items():
        if not name.startswith('state_head.'):
            assert torch.equal(value, shared[name]), 'RUNTIME_INITIALIZATION_CHANGED:'+name
    return net


def inputs():
    data = c.read(HERE.parent/'multifamily_v7/DATA.json')
    natural = c.read(HERE.parent/'natural_transfer_v9/DATA.json')
    caches = {}
    for name, folder in [('special', 'multifamily_v7'), ('natural', 'natural_transfer_v9')]:
        run = HERE.parent/folder/'run_001'
        assert c.sha(run/'FEATURES.pt') == c.read(run/'FEATURE_RESULT.json')['file_sha256']
        caches[name] = {k: v.float() for k, v in torch.load(run/'FEATURES.pt', map_location='cpu', weights_only=True).items()}
    return data, natural, caches


def check():
    data, _, caches = inputs()
    net = initialize(1209, data)
    batch = train.special.tensors(data['families'][0], 'cpu')
    initial = c.model_identity(net)['sha256']
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = train.special.losses(net, caches['special'], batch, 'B2')
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
        optimizer.step()
    assert c.model_identity(net)['sha256'] != initial
    gradient = train.gradient_audit(net, caches['special'], batch, 'B2')
    c.write(HERE/'STRONG_STATE_CPU_TEST.json', dict(passed=True, disposable_optimizer_updates=2,
        shared_runtime_initialization_equal=True, exact_state_bits_unchanged=True,
        state_head_parameters=sum(p.numel() for p in net.state_head.parameters()),
        outcome_head_parameters=sum(p.numel() for p in net.result_head.parameters()),
        gradient=gradient, gpu_hours=0, base_updates=0, new_qwen_forwards=0), True)


def run():
    config = c.read(HERE/'STRONG_STATE_PROTOCOL.json')
    assert c.read(HERE/'STRONG_STATE_CPU_TEST.json')['passed']
    for path, expected in config['source_hashes'].items():
        assert c.sha(train.LINE/path) == expected, 'SOURCE_CHANGED:'+path
    data, natural, caches = inputs()
    batches = [train.special.tensors(f, 'cpu') for f in data['families']]
    folder = HERE/'strong_state_run_001'
    folder.mkdir(exist_ok=False)
    began = time.monotonic()
    c.write(folder/'RUNTIME_IDENTITY.json', dict(device='cpu', threads=4, torch_version=torch.__version__,
        protocol_sha256=c.sha(HERE/'STRONG_STATE_PROTOCOL.json'), gpu_hours=0,
        base_updates=0, new_qwen_forwards=0, future_query_in_runtime=False), True)
    results = {}
    for seed in config['seeds']:
        net = initialize(seed, data)
        initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
        initial_sha = c.model_identity(net)['sha256']
        torch.save(initial, folder/f'INITIAL_{seed}.pt')
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        schedule = c.read(HERE.parent/'natural_transfer_v9/run_001'/f'SCHEDULE_{seed}.json')
        assert len(schedule['family_indices']) == len(schedule['ordinary_indices']) == 600
        key = f'B2_MLP_{seed}'
        for step, (family, ordinary) in enumerate(zip(schedule['family_indices'], schedule['ordinary_indices']), 1):
            optimizer.zero_grad(set_to_none=True)
            special, stats, _ = train.special.losses(net, caches['special'], batches[family], 'B2')
            batch = train.ordinary_batch([natural['records'][i] for i in ordinary], 'cpu')
            ordinary_loss, _ = train.ordinary_loss(net, caches['natural'], batch)
            loss = special+ordinary_loss
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
            optimizer.step()
            c.append(folder/f'{key}_STEPS.jsonl', dict(step=step, family_index=family, ordinary_indices=ordinary,
                loss=float(loss.detach()), ordinary_ce=float(ordinary_loss.detach()), **stats))
            c.write(folder/'PROGRESS.json', dict(unix=time.time(), seed=seed, arm='B2_MLP', step=step, total=600))
            if step == 2:
                c.write(folder/f'{key}_GRADIENT.json', train.gradient_audit(net, caches['special'], batches[schedule['family_indices'][0]], 'B2'), True)
        measured = train.special.evaluate(net, caches['special'], batches, 'B2')
        measured['branch_actions'] = train.branch_metrics.evaluate(net, caches['special'], data,
            c.read(HERE.parent/'cost_teacher_v13/ACTOR_ADMISSION.json'))
        measured['ordinary'] = train.ordinary_evaluate(net, caches['natural'], natural['records'])
        final = {k: v.detach().clone() for k, v in net.state_dict().items()}
        changes = {k: not torch.equal(v, initial[k]) for k, v in final.items()}
        for name in ('writer.weight', 'recurrent.weight', 'action.0.weight', 'action.2.weight', 'state_head.0.weight', 'state_head.2.weight'):
            assert changes[name], 'TRAINED_PARAMETER_UNCHANGED:'+name
        assert not any(changes[k] for k in changes if k.startswith('result_head.'))
        checkpoint = folder/f'{key}_MEMORY.pt'
        torch.save(final, checkpoint)
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        assert train.branch_metrics.evaluate(net, caches['special'], data,
            c.read(HERE.parent/'cost_teacher_v13/ACTOR_ADMISSION.json')) == measured['branch_actions']
        measured.update(seed=seed, arm='B2_MLP', updates=600, initial_state_sha256=initial_sha,
            final_state_sha256=c.model_identity(net)['sha256'], parameter_changed=changes,
            checkpoint_sha256=c.sha(checkpoint), checkpoint_branch_actions_read_back=True)
        c.write(folder/f'{key}_RESULT.json', measured, True)
        results[key] = measured
    c.write(folder/'RESULT.json', dict(status='STRONG_EXACT_STATE_CONTROL_COMPLETE', runs=results,
        optimizer_updates=1800, cpu_wall_seconds=time.monotonic()-began, gpu_hours=0,
        original_training_admission=False, closed_loop_gain_measured=False, publication_ready=False), True)


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1:] == ['--test']:
        check()
    else:
        assert not sys.argv[1:]
        run()
