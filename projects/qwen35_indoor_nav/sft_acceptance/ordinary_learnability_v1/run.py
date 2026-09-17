"""One bounded single-GPU diagnostic, using the original policy and data decoder."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_state(gpu):
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x', '-i', str(gpu)], text=True, timeout=15))
    card = root.find('gpu')
    return dict(uuid=card.findtext('uuid'), memory_mib=float(card.findtext('fb_memory_usage/used').split()[0]),
                pids=[int(p.findtext('pid')) for p in card.findall('processes/process_info')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['current', 'coverage'], default='current')
    args = parser.parse_args()
    out = HERE / args.arm
    out.mkdir()  # Exclusive attempt: never overwrite a completed/failed run.
    started = time.monotonic()
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, _: stop.append(s))
    before = gpu_state(4)
    assert not before['pids'] and before['memory_mib'] < 128, 'GPU4_NOT_EMPTY'
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CONDA_PREFIX'):
        os.environ.pop(key, None)
    os.environ.update(CUDA_VISIBLE_DEVICES=before['uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for key in ('TMPDIR', 'TMP', 'TEMP', 'HF_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME',
                'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        path = HERE / 'cache' / args.arm / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    save(out / 'START.json', dict(pid=os.getpid(), unix=time.time(), physical_gpu=4, before=before,
        wall_budget_seconds=2700, source_hashes={str(p.relative_to(LINE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE/'run.py',HERE/'prepare.py',HERE/'PLAN_ZH.md',HERE/'INPUTS.jsonl',HERE/'SUPERVISION.jsonl')}))
    r = load('learnability_runtime', HERE.parent / 'ordinary_history8_paired_train_r1/runtime.py')
    r.configure()
    import numpy as np
    import torch
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(25*1024**3/torch.cuda.get_device_properties(0).total_memory)
    preparation = json.loads((HERE / 'PREPARATION.json').read_text())
    assert preparation['conflicting_inputs'] == 0, 'UNEXPLAINED_INPUT_CONFLICTS'
    assert r.sha(HERE/'INPUTS.jsonl') == preparation['input_sha256']
    assert r.sha(HERE/'SUPERVISION.jsonl') == preparation['label_sha256']
    inputs = [json.loads(x) for x in (HERE / 'INPUTS.jsonl').read_text().splitlines()]
    labels = [json.loads(x) for x in (HERE / 'SUPERVISION.jsonl').read_text().splitlines()]
    assert all(a['input_id'] == b['input_id'] for a,b in zip(inputs,labels)) and len(inputs)==len(labels)==384
    rows = [json.loads(x) for x in (LINE / 'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/TRAINING_INDEX.jsonl').read_text().splitlines()]
    model, policy, initial = r.initialize()
    initial_state = initial['trainable']
    del initial
    assert args.arm == 'current', 'Coverage implementation requires completed current-arm evidence'
    data = load('learnability_original_data', HERE.parent / 'ordinary_sync_recovery_v1/data.py')
    store = data.SampleStore(rows)
    samples = [dict(record_idx=x['record_idx'],t=x['t'],target=y['target'],weight=1.) for x,y in zip(inputs,labels)]
    dataset = model.DecisionDataset(samples, store, policy.processor)
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id, policy.exec_sid,
                                 policy.query_sid, policy.base.config.image_token_id)
    window_module = load('learnability_original_window', LINE / 'closed_loop_bench/r2r_ce_tiny_v1/common.py')
    # Preprocess the finite repeated small set once on CPU; never cache learned features.
    encoded = []
    for i, sample in enumerate(samples):
        item = store.get(sample['record_idx'], sample['t'])
        assert item['target'] == sample['target']
        identity = [item['instruction'], [hashlib.sha256(image.tobytes()).hexdigest() for image in item['images']], item['executed']]
        assert hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == inputs[i]['input_id']
        encoded.append(dataset[i])
    # Exercise the actual online Window.receive on one frozen example per class/group.
    checked = set()
    for i, sample in enumerate(samples):
        kind = (inputs[i]['group'],sample['target'])
        if kind in checked:
            continue
        checked.add(kind)
        record = store._cache[sample['record_idx']]
        window = window_module.Window()
        for t in range(sample['t']+1):
            im = store._data.load_rgb(store._data._relative(record.rgb_root, record._refs[t]))
            payload = dict(done=False, rgb=base64.b64encode(im.tobytes()).decode())
            if t == 0:
                payload['instruction'] = record.instruction
            window.receive(payload, executed=None if t==0 else record.actions[t-1])
        class OnlineStore:
            def get(self, idx, t):
                return window.item()
        online = model.DecisionDataset([sample], OnlineStore(), policy.processor)[0]
        for key, value in online.items():
            assert torch.equal(value, encoded[i][key]) if isinstance(value, torch.Tensor) else value == encoded[i][key]
    batches = [collate([item]) for item in encoded]
    del encoded
    train_ids = [i for i,x in enumerate(inputs) if x['group']=='train']
    check_ids = [i for i,x in enumerate(inputs) if x['group']=='check']
    policy.eval()
    def forward(i):
        batch = batches[i]
        assert batch['action_index'].tolist()==[[0,int(batch['attention_mask'].sum())-1]]
        return policy.forward_batch(**{k:v.to('cuda') for k,v in batch.items() if k not in ('targets','weights')})
    with torch.inference_mode():
        a = forward(train_ids[0]); b = forward(train_ids[0])
        assert torch.equal(a,b), 'REPEAT_FORWARD_PARITY'
    save(out / 'INTERFACE.json', dict(status='PASS', pixel_and_target_bindings=384,
        online_window_and_encoding_cases=len(checked), repeated_forward_max_abs=float((a-b).abs().max()),
        trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
        trainable_tensors=sum(p.requires_grad for p in policy.parameters()),
        actual_input='Original DecisionDataset/make_collate/forward_batch; no privileged fields',
        preparation_sha256=r.sha(HERE/'PREPARATION.json'), initializer_sha256=r.BRIDGE_SHA))
    del a,b
    torch.manual_seed(1209); np.random.seed(1209); random.seed(1209)
    trainable = {n:p for n,p in policy.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(trainable.values()),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    targets = [torch.tensor([s['target']],device='cuda') for s in samples]
    evaluations = []
    completed = 0
    reads = 0
    reason = None

    def limits():
        if stop:
            return 'SIGNAL'
        if time.monotonic()-started >= 2550:  # Reserve final evaluation/checkpoint time.
            return 'WALL_BUDGET'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 64*1024**3, 'RSS_BUDGET'
        assert torch.cuda.max_memory_reserved() <= 25*1024**3, 'MODEL_MEMORY_BUDGET'
        return None

    def evaluate():
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
        policy.eval()
        result = dict(updates=completed, training_reads=reads, elapsed_seconds=time.monotonic()-started, groups={})
        try:
            with torch.inference_mode():
                for name, indices in (('train',train_ids),('check',check_ids)):
                    matrix = torch.zeros((4,4),dtype=torch.int64,device='cuda')
                    ce = torch.zeros((),device='cuda')
                    per_scene = {}
                    for i in indices:
                        logits = forward(i)
                        loss = torch.nn.functional.cross_entropy(logits, targets[i])
                        y, pred = samples[i]['target'], int(logits.argmax(-1))
                        matrix[y,pred] += 1; ce += loss
                        scene = per_scene.setdefault(inputs[i]['scene'],dict(n=0,correct=0))
                        scene['n'] += 1; scene['correct'] += y==pred
                    result['groups'][name] = dict(n=len(indices),ce=float(ce/len(indices)),
                        accuracy=float(matrix.diag().sum()/len(indices)),
                        recall=(matrix.diag()/matrix.sum(1)).cpu().tolist(),confusion=matrix.cpu().tolist(),scenes=per_scene)
            evaluations.append(result)
            save(out / ('EVAL_%04d.json'%completed), result)
            print(json.dumps(dict(event='FIXED_EVAL',**result)),flush=True)
        finally:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3])
            policy.train()

    evaluate()
    order = []
    generator = random.Random(1209)
    while len(order) < 12800:
        epoch = train_ids.copy();generator.shuffle(epoch);order.extend(epoch)
    first_update = None
    last_resource = time.monotonic()
    for step in range(400):
        reason = limits()
        if reason:
            break
        if time.monotonic()-last_resource >= 15:
            card = gpu_state(before['uuid'])
            assert set(card['pids']) <= {os.getpid()}, 'FOREIGN_GPU_CONTEXT'
            assert card['memory_mib'] <= 28*1024, 'TOTAL_GPU_MEMORY_BUDGET'
            last_resource = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.
        for i in order[step*32:(step+1)*32]:
            logits = forward(i)
            loss = torch.nn.functional.cross_entropy(logits, targets[i])/32
            loss.backward();reads += 1;ce_sum += float(loss.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values()), 'INVALID_GRADIENT'
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True)
        optimizer.step();completed += 1
        if completed == 1:
            first_update = {n:dict(grad_norm=float(p.grad.norm()),update_l2=float((p.detach().cpu()-initial_state[n]).norm())) for n,p in trainable.items()}
            save(out/'FIRST_UPDATE.json',first_update)
        if completed % 5 == 0 or completed == 1:
            event = dict(updates=completed,training_reads=reads,train_ce=ce_sum,grad_norm=float(norm),
                         elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save(out/'PROGRESS.json',event)
            print(json.dumps(dict(event='TRAIN',**event)),flush=True)
        if completed in (100,200,400):
            evaluate()
    if evaluations[-1]['updates'] != completed:
        evaluate()
    changes = {n:float((p.detach().cpu()-initial_state[n]).norm()) for n,p in trainable.items()}
    finite = all(torch.isfinite(p).all() for p in trainable.values()) and all(torch.isfinite(v).all() for slot in optimizer.state.values() for v in slot.values() if isinstance(v,torch.Tensor))
    assert finite and all(value>0 for value in changes.values()), 'INVALID_PARAMETER_UPDATES'
    metrics = evaluations[-1]['groups']['train']
    passed = completed==400 and metrics['accuracy']>=.95 and min(metrics['recall'])>=.90
    checkpoint = out/'final.pt'
    torch.save(dict(trainable=model.trainable_state(policy),updates=completed,arm=args.arm,
        input_sha256=preparation['input_sha256'],initializer_sha256=r.BRIDGE_SHA),checkpoint)
    save(out/'RESULT.json',dict(status='COMPLETE' if completed==400 else 'RESOURCE_CENSORED',
        learnability_pass=passed if completed==400 else None,updates=completed,training_reads=reads,
        stop_reason=reason,evaluations=evaluations,parameter_update_l2=changes,
        all_parameters_and_optimizer_finite=True,wall_seconds=time.monotonic()-started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        checkpoint=str(checkpoint.relative_to(LINE)),checkpoint_sha256=r.sha(checkpoint),navigation_gain=None))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # Keep the traceback; external process exit releases this run's CUDA context.
        arm = sys.argv[sys.argv.index('--arm')+1] if '--arm' in sys.argv else 'current'
        if (HERE/arm).is_dir() and not (HERE/arm/'FAILURE.json').exists():
            save(HERE/arm/'FAILURE.json',dict(unix=time.time(),error=traceback.format_exc()))
        print(traceback.format_exc(),file=sys.stderr,flush=True)
        raise
