"""Freeze the trained ORIGINAL action/memory paths; update only events for200steps."""
import argparse
import json
import random
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import *
base_common=load('teacher_source_common',PARENT/'common.py');sys.modules['common']=base_common
source_train=load('teacher_checkpoint_io',PARENT/'train.py')
EvidencePolicy=source_train.EvidencePolicy
objective=source_train.objective
verify_binding=base_common.verify_binding
sys.path.insert(0,str(HERE))


def configure_trainable(net):
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith('events.'))

def frozen_state(net):
    return {name:c.tensor_identity(value.detach().cpu())['sha256']
            for name,value in net.state_dict().items() if not name.startswith('events.')}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def save(path, net, optimizer, step, binding):
    atomic_torch(path, dict(model=net.state_dict(), optimizer=optimizer.state_dict(), step=step,
                            cursor=step, binding=binding, rng=rng_state()))
    immutable(path.with_suffix('.json'), dict(sha256=sha(path), step=step, binding=binding))


def restore(path, net, optimizer, binding):
    meta = read(path.with_suffix('.json'))
    if meta['binding'] != binding or sha(path) != meta['sha256']:
        raise ValueError('RESUME_IDENTITY_CHANGED')
    # Only a hash-bound, locally produced checkpoint is accepted.
    record = torch.load(path, map_location=next(net.parameters()).device, weights_only=False)
    if record['binding'] != binding or record['step'] != record['cursor']:
        raise ValueError('RESUME_CURSOR_CHANGED')
    net.load_state_dict(record['model']); optimizer.load_state_dict(record['optimizer'])
    r = record['rng']; random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
    return record['step']


def main(args):
    run, output = args.run.resolve(), args.output.resolve()
    verify_lock(read(run/'SOURCE_LOCK.json'))
    bound = verify_binding(run); cfg = read(run / 'PROTOCOL.json')
    if args.seed not in cfg['seeds'] or not 1 <= args.updates <= cfg['steps']:
        raise ValueError('UNREGISTERED_TRAINING_BUDGET')
    if args.device == 'cpu' and args.updates > cfg['cpu_smoke_updates']:
        raise ValueError('CPU_SMOKE_BUDGET; full study belongs to the next GPU stage')
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir() or (output / 'RESULT.json').exists():
        raise ValueError('NO_INCOMPLETE_RUN_TO_RESUME')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.mode != 'REPAIR': raise ValueError('REFERENCE_MUST_NOT_TRAIN')
    net = EvidencePolicy(args.seed, 'MONOTONIC').to(args.device)
    source = Path(cfg['source_original_run'])/'train'/f'ORIGINAL_{args.seed}'
    origin = read(source/'RESULT.json')
    if sha(source/'FINAL.pt') != origin['checkpoint_sha256']: raise ValueError('ORIGINAL_CHECKPOINT_CHANGED')
    net.load_state_dict(torch.load(source/'FINAL.pt',map_location=args.device,weights_only=True))
    configure_trainable(net)
    frozen_before = frozen_state(net)
    opt = torch.optim.AdamW(net.events.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    initial = c.model_identity(net)['sha256']
    if initial != origin['final']: raise ValueError('LOADED_ORIGINAL_STATE_MISMATCH')
    if not (output/'INITIAL.pt').exists():atomic_torch(output/'INITIAL.pt',net.state_dict())
    else:
        saved_initial=torch.load(output/'INITIAL.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(saved_initial[k],v.cpu()) for k,v in net.state_dict().items()):raise ValueError('INITIALIZATION_CHANGED')
    immutable(output/'INITIAL.json',dict(seed=args.seed,state_sha256=initial,source=str(source/'FINAL.pt'), source_sha256=origin['checkpoint_sha256']))
    binding = dict(inputs=digest(bound), mode=args.mode, seed=args.seed, initial=initial,
                   device=args.device, target_updates=args.updates)
    cursor = 0
    checkpoints = sorted(p for p in output.glob('attempt_*/STEP_*.pt') if p.with_suffix('.json').exists())
    if args.resume:
        if not checkpoints:
            raise ValueError('NO_SEALED_CHECKPOINT')
        latest = max(checkpoints, key=lambda p: read(p.with_suffix('.json'))['step'])
        cursor = restore(latest, net, opt, binding)
    attempt = output / f'attempt_{len(list(output.glob("attempt_*")))+1:03d}'; attempt.mkdir()
    if cursor==0 and not checkpoints:save(attempt/'STEP_0000.pt',net,opt,0,binding)
    data = read(run/'DATA.json')
    fit = [f for f in data['families'] if f['split'] == 'FIT']; families = {f['family_id']: f for f in fit}
    w = objective.weights(fit); immutable(output / 'FIT_WEIGHTS.json', w)
    schedule = read(run / 'SCHEDULES.json')[str(args.seed)]
    cache = {k:v.float().to(args.device) for k,v in torch.load(run/'features/FEATURES.pt', map_location='cpu', weights_only=True).items()}
    natural = read(run/'ORDINARY_DATA.json')
    oldpath = run/'features/ORDINARY_FEATURES.pt'
    ordinary = {k:v.float().to(args.device) for k,v in torch.load(oldpath, map_location='cpu', weights_only=True).items()}
    event_module=load('event_loss_repair',HERE/'event_loss.py')
    event_pool=event_module.Pool(read(run/'PRESERVATION_POOL.json')['records'])
    began = time.monotonic(); parameter_count = sum(p.numel() for p in net.parameters())
    for i in range(cursor, args.updates):
        item = schedule[i]
        if item['family'] not in families or any(natural['records'][j]['partition'] != 'fit' for j in item['ordinary']):
            raise ValueError('NONFIT_OPTIMIZATION')
        b = objective.batch(families[item['family']], args.device)
        opt.zero_grad(set_to_none=True)
        loss, stats, detail = objective.losses(net, cache, b, w)
        stats['original_event_bce']=stats['event_bce']
        stats['event_draws']=0
        if args.mode=='REPAIR':
            replacement=event_pool.sample_loss(net,cache,args.seed,i,cfg['event_batch_size'])
            loss=loss-detail['event_loss']+replacement
            stats['event_bce']=float(replacement.detach());stats['event_draws']=cfg['event_batch_size']
        bc = objective.ordinary_loss(net, ordinary, [natural['records'][j] for j in item['ordinary']])
        total = loss + bc
        if not torch.isfinite(total):
            raise ValueError('NONFINITE_LOSS')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise ValueError('NONFINITE_GRADIENT')
        if any(p.grad is not None for n,p in net.named_parameters() if not n.startswith('events.')):raise ValueError('GRADIENT_ESCAPED_EVENT_MODULE')
        norms = {name:float(p.grad.norm()) for name,p in net.named_parameters() if p.grad is not None}
        opt.step()
        c.append(attempt/'STEPS.jsonl', dict(step=i+1, family=item['family'], split='FIT', loss=float(total.detach()),
                    ordinary_ce=float(bc.detach()), gradients=norms, **stats))
        write(output/'PROGRESS.json', dict(step=i+1, target=args.updates, mode=args.mode, seed=args.seed, device=args.device))
        if (i+1) % cfg['checkpoint_every'] == 0 or i+1 == args.updates or args.device == 'cpu':
            save(attempt/f'STEP_{i+1:04d}.pt', net, opt, i+1, binding)
    if not (output/'FINAL.pt').exists():atomic_torch(output/'FINAL.pt', net.state_dict())
    else:
        saved_final=torch.load(output/'FINAL.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(saved_final[k],v.cpu()) for k,v in net.state_dict().items()):raise ValueError('UNSEALED_FINAL_DIFFERS_FROM_RESTORED_STATE')
    if frozen_state(net)!=frozen_before: raise ValueError('FROZEN_ACTION_MEMORY_PARAMETERS_CHANGED')
    final = c.model_identity(net)['sha256']
    record = dict(status='CPU_SMOKE_COMPLETE' if args.device == 'cpu' else 'TRAINING_COMPLETE',
                  mode=args.mode, seed=args.seed, updates=args.updates, initial=initial, final=final,
                  parameters_changed=initial!=final, parameters=parameter_count, base_updates=0,
                  frozen_before=frozen_before,frozen_after=frozen_state(net),frozen_unchanged=True,
                  trainable_parameters=[n for n,p in net.named_parameters() if p.requires_grad],
                  source_original=origin['final'],trainable_count=sum(p.numel() for p in net.parameters() if p.requires_grad),
                  base_loaded=False, cuda_initialized=torch.cuda.is_initialized(), seconds=time.monotonic()-began,
                  checkpoint_sha256=sha(output/'FINAL.pt'), navigation_gain=None, evaluation_episodes=0,
                  note='CPU updates are interface checks, excluded from full study; same MONOTONIC initialization/architecture, Starts from trained ORIGINAL; only event MLP updates, all original action labels retained. Frozen weights do not guarantee identical actions because predicted states can change.')
    write(output/'RESULT.json', record, True); print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('ORIGINAL','REPAIR'), required=True); parser.add_argument('--seed', type=int, default=1209)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
