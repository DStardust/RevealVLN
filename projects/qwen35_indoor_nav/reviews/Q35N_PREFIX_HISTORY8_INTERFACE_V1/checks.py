import gc
import json
from pathlib import Path
import time

def run(model, policy, collate, forward, c, fingerprint, initial_sha):
    import torch
    parent = c.LINE / 'sft_acceptance/ordinary_prefix_history8_v1'
    fixed = c.load('old32_numerical_regression', c.LINE / 'reviews/Q35N_STOP_FEATURE_PREFIX_DIAGNOSTIC_V2/checks.py')
    old = fixed.check(model, policy, collate, forward, c, 'entry')
    assert old['max_abs_old'] <= 1e-5 and old['original_argmax_same'], 'ORIGINAL_ENTRY_REGRESSION'
    data = c.load('history8_ordinary_view', parent / 'data.py')
    samples = json.loads((parent / 'DIAGNOSTIC_SAMPLES.json').read_text())
    assert len(samples) == 24
    rows, report = data.load_rows()
    stores = {'two_rgb': data.base['SampleStore'](rows), 'prefix_eight_rgb': data.SampleStore(rows)}
    result = {}
    for mode, store in stores.items():
        dataset = model.DecisionDataset(samples, store, policy.processor)
        encoded = [dataset[i] for i in range(len(samples))]
        assert all(int(x['image_grid_thw'].shape[0]) == (8 if mode == 'prefix_eight_rgb' else (1 if samples[i]['t'] == 0 else 2)) for i,x in enumerate(encoded))
        policy.eval()
        infer_seconds, logits = [], []
        with torch.inference_mode():
            for item in encoded:
                tick = time.perf_counter()
                logits.append(forward([item]).cpu())
                infer_seconds.append(time.perf_counter() - tick)
        z = torch.cat(logits)
        assert z.shape == (24, 4) and torch.isfinite(z).all()
        c.write(c.HERE / 'run_001' / (mode + '_LOGITS.json'), dict(samples=samples, logits=z.tolist(), inference_seconds=infer_seconds), True)
        policy.train()
        train_seconds, losses, gradient_rows = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            policy.zero_grad(set_to_none=True)
            batch = collate(encoded[i*4:(i+1)*4])
            targets = batch.pop('targets').to('cuda:0')
            weights = batch.pop('weights').to('cuda:0')
            batch = {key:value.to('cuda:0') for key,value in batch.items()}
            torch.cuda.synchronize()
            tick = time.perf_counter()
            output = policy.forward_batch(**batch)
            loss = (torch.nn.functional.cross_entropy(output, targets, reduction='none') * weights).sum() / weights.sum()
            assert torch.isfinite(loss)
            loss.backward()
            torch.cuda.synchronize()
            train_seconds.append(time.perf_counter() - tick)
            losses.append(float(loss.detach()))
            norms = {}
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None, 'FROZEN_BACKBONE_GRADIENT'
                else:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'MISSING_OR_NONFINITE_GRADIENT:' + name
                    norms[name] = float(parameter.grad.detach().float().norm())
            gradient_rows.append(norms)
        result[mode] = dict(single_inference_seconds=infer_seconds,
            cold_forward_backward_seconds=train_seconds[0],
            warm_forward_backward_seconds=train_seconds[1:],
            warm_training_decisions_per_second=12/sum(train_seconds[1:]),
            microbatch=4, diagnostic_ce=losses, gradient_norms=gradient_rows,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        policy.zero_grad(set_to_none=True); policy.eval()
        del encoded, dataset, batch, output, loss, z, logits
        gc.collect(); torch.cuda.empty_cache()
        assert fingerprint() == initial_sha, 'PARAMETERS_CHANGED_WITHOUT_OPTIMIZER'
        c.write(c.HERE / 'PROGRESS.json', dict(status='DIAGNOSING',unix=time.time(),completed_mode=mode,parameter_updates=0), False)
    assert fingerprint() == initial_sha
    c.write(c.HERE / 'RESULT.json', dict(status='PASS_INTERFACE_ONLY',unix=time.time(),
        old_entry_max_abs=old['max_abs_old'],old_entry_argmax_same=old['original_argmax_same'],
        modes=result,source_snapshot=report,trainable_fingerprint=initial_sha,
        parameters_exact_unchanged=True,forward_decisions=144,backward_calls=8,
        parameter_updates=0,simulator_actions=0,navigation_gain=False,
        next_action='Freeze ordinary-only 2-vs-8 history training control and budget using measured cost'), True)
    print(json.dumps(dict(status='PASS_INTERFACE_ONLY',forward_decisions=144,backward_calls=8,parameter_updates=0)), flush=True)

