"""Matched event-only leave-one-FIT-house-out runs; no held-out policy training."""
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
from event_loss import Pool, ROLES, original_mass, predictions, metrics
from train_one import save, restore, verify_binding


def execute(run, arm, fold):
    cfg = config(run); bound = verify_binding(run)
    if arm not in cfg['arms'] or fold not in range(4):
        raise ValueError('UNREGISTERED_PROBE')
    seed = cfg['event_probe_seed']; steps = cfg['event_probe_steps']
    target = run/'event_probe'/f'{arm}_{fold}'
    target.mkdir(parents=True, exist_ok=True)
    if (target/'RESULT.json').exists():
        result = read(target/'RESULT.json')
        if result['steps'] != steps or sha(target/'FINAL.pt') != result['final_file_sha256']:
            raise ValueError('COMPLETED_PROBE_CHANGED')
        return
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = make_head(f'{arm}_{seed}').to('cuda:0')
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.events.parameters():
        parameter.requires_grad_(True)
    initial = c.model_identity(net)['sha256']
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if not k.startswith('events.')}
    if not (target/'INITIAL.pt').exists():
        atomic_torch(target/'INITIAL.pt', net.state_dict())
    houses = read(run/'EVENT_AUDIT.json')['fold_houses']; held = houses[fold]
    all_records = read(run/'EVENT_POOL.json')['records']
    train = [r for r in all_records if r['house'] != held]
    check = [r for r in all_records if r['house'] == held]
    if {r['feature_index'] for r in train} & {r['feature_index'] for r in check}:
        raise ValueError('HELD_HOUSE_INPUT_LEAK')
    families = [f for f in read(run/'DATA.json')['families'] if f['split'] == 'FIT' and f['house'] != held]
    schedule = read(run/'SCHEDULES.json')[str(seed)]
    original, exposure = original_mass(families, train, schedule)
    weights = original if arm == 'ORIGINAL' else Pool(train).mass()
    binding = dict(inputs=digest(bound), arm=arm, fold=fold, held=held, steps=steps,
                   initial=initial, loss_mass_sha256=digest(weights), train_ids=digest(train), check_ids=digest(check))
    immutable(target/'BINDING.json', binding)
    immutable(target/'SUPERVISION.json', dict(exposure=exposure, n_train=len(train), n_held_out=len(check),
        mass=weights, train_keys=[[r['feature_index'], r['role']] for r in train],
        frozen_normalization=True, policy_training=False, dev_access=False,
        only_trainable=[k for k, v in net.named_parameters() if v.requires_grad]))
    opt = torch.optim.AdamW(net.events.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    checkpoints = [p for p in target.glob('attempt_*/STEP_*.pt') if p.with_suffix('.json').exists()]
    cursor = restore(max(checkpoints, key=lambda p: read(p.with_suffix('.json'))['step']), net, opt, binding) if checkpoints else 0
    attempt = target/f'attempt_{len(list(target.glob("attempt_*")))+1:03d}'; attempt.mkdir()
    if not checkpoints:
        save(attempt/'STEP_0000.pt', net, opt, 0, binding)
    cache = {k: v.float().to('cuda:0') for k, v in torch.load(run/'features/FEATURES.pt', weights_only=True, map_location='cpu').items()}
    indices = torch.tensor([r['feature_index'] for r in train], device='cuda:0')
    roles = torch.tensor([ROLES.index(r['role']) for r in train], device='cuda:0')
    truth = torch.tensor([r['target'] for r in train], dtype=torch.float32, device='cuda:0')
    mass = torch.tensor(weights, device='cuda:0')
    with torch.no_grad():
        normalized = net.core.feature_norm(cache['features'][indices]).detach()
    began = time.monotonic()
    for step in range(cursor, steps):
        opt.zero_grad(set_to_none=True)
        logits = net.events(normalized).gather(1, roles[:, None]).squeeze(1)
        loss = (F.binary_cross_entropy_with_logits(logits, truth, reduction='none') * mass).sum()
        if not torch.isfinite(loss):
            raise ValueError('NONFINITE_PROBE_LOSS')
        loss.backward()
        norms = {k: float(v.grad.norm()) for k, v in net.events.named_parameters()}
        if any(not torch.isfinite(v.grad).all() for v in net.events.parameters()):
            raise ValueError('NONFINITE_PROBE_GRADIENT')
        opt.step()
        append(attempt/'STEPS.jsonl', dict(step=step+1, loss=float(loss.detach()), gradients=norms,
                                          supervised_labels=len(train), base_updates=0))
        write(target/'PROGRESS.json', dict(step=step+1, target=steps, arm=arm, held_house=held, stage='event_probe'))
        if (step+1) % cfg['checkpoint_every'] == 0 or step+1 == steps:
            save(attempt/f'STEP_{step+1:04d}.pt', net, opt, step+1, binding)
    if any(not torch.equal(v, net.state_dict()[k].cpu()) for k, v in frozen.items()):
        raise ValueError('NON_EVENT_PARAMETERS_UPDATED')
    with torch.no_grad():
        probabilities = predictions(net, cache, check).sigmoid().cpu().tolist()
    immutable(target/'PREDICTIONS.json', [dict(feature_index=r['feature_index'], role=r['role'], house=r['house'],
        target=r['target'], probability=p) for r, p in zip(check, probabilities)])
    if not (target/'FINAL.pt').exists():
        atomic_torch(target/'FINAL.pt', net.state_dict())
    result = dict(status='EVENT_PROBE_COMPLETE', arm=arm, fold=fold, held_house=held, seed=seed, steps=steps,
        initial=initial, final=c.model_identity(net)['sha256'], final_file_sha256=sha(target/'FINAL.pt'),
        prediction_sha256=sha(target/'PREDICTIONS.json'), binding_sha256=sha(target/'BINDING.json'),
        metrics=metrics(probabilities, check), seconds=time.monotonic()-began, base_updates=0, base_loaded=False,
        full_policy_updates=0, held_house_excluded_from_all_training=True, dev_access=False,
        frozen_parameters_unchanged=True, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if result['initial'] == result['final']:
        raise ValueError('NO_EVENT_PARAMETER_UPDATE')
    immutable(target/'RESULT.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    run = Path(sys.argv[1])
    for tag in os.environ['B2_TRAIN_TAGS'].split(','):
        arm, fold = tag.rsplit('_', 1)
        execute(run, arm, int(fold))
