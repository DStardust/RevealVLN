"""Frozen-panel evaluation for ordinary baseline v3 checkpoints.

Teacher-forced forward over PANEL_FIT.jsonl with the v3 batching path.
Reports CE, per-class recall, macro recall, prediction/target distributions,
and the always-forward baseline delta. No training, no state, no DEV.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SPEED = LINE / 'sft_acceptance/ordinary_speedup_10x_v1'
FLA_DEPS = SPEED / 'official_fla_0_5_2/deps'
EINOPS_DEPS = SPEED / 'official_einops_0_8_1/deps'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-protocol', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'OFFLINE_FLAGS')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_CONFIG')
    spec = json.loads(args.eval_protocol.read_text())
    require(spec['id'] == 'Q35N_V3_PANEL_EVAL_V1', 'EVAL_PROTOCOL_ID')
    for name, digest in spec['code_sha256'].items():
        require(sha256(HERE / name) == digest, 'CODE_CHANGED:' + name)
    require(sha256(HERE / 'PANEL_FIT.jsonl') == spec['panel_sha256'], 'PANEL_BINDING')

    if spec['kernel'] == 'fla':
        sys.path.insert(0, str(EINOPS_DEPS))
        sys.path.insert(0, str(FLA_DEPS))
        import fla.ops.gated_delta_rule  # noqa: F401
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (c.cell_contents for c in bound.__closure__)))
    resolved = getattr(cells.get('implementation'), '__module__', '') or ''
    if spec['kernel'] == 'fla':
        require(resolved.startswith('fla.'), 'FLA_NOT_RESOLVED')
    else:
        require(resolved.endswith('modeling_qwen3_5'), 'FLA_LEAKED')

    data = load_module('q35n_v3_eval_data', HERE / 'data.py')
    model = load_module('q35n_v3_eval_model', HERE / 'model.py')
    rows, report = data.load_rows()
    require(report['training_index_sha256'] == spec['snapshot']['training_index_sha256'],
            'SNAPSHOT_BINDING')
    samples = data.load_sample_index(HERE / 'SAMPLE_INDEX.jsonl', spec['sample_index_sha256'],
                                     sum(r['decisions'] for r in rows))

    device = torch.device('cuda', 0)
    torch.cuda.set_per_process_memory_fraction(
        spec['max_gpu_memory_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    policy = model.build_policy(spec['seed'])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_trainable(policy, state['trainable'])
    policy.eval()

    panel = []
    for line in (HERE / 'PANEL_FIT.jsonl').read_text().splitlines():
        if line.strip():
            r, t = json.loads(line)
            panel.append((r, t))
    lookup = {}
    wanted = set(panel)
    for s in samples:
        key = (s['record_idx'], s['t'])
        if key in wanted:
            lookup[key] = s
    require(len(lookup) == len(panel), 'PANEL_LOOKUP_INCOMPLETE')
    ordered = [lookup[k] for k in panel]
    # length-sorted packs under the registered eval token budget
    ordered.sort(key=lambda s: s['est'])
    packs, current, current_tokens = [], [], 0
    for s in ordered:
        if current and current_tokens + s['est'] > spec['eval_max_tokens']:
            packs.append(current)
            current, current_tokens = [], 0
        current.append(s)
        current_tokens += s['est']
    if current:
        packs.append(current)

    store = data.SampleStore(rows)
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id, policy.exec_sid,
                                 policy.query_sid, policy.base.config.image_token_id)
    confusion = [[0] * 4 for _ in range(4)]
    ce_sum = 0.0
    n = 0
    with torch.no_grad():
        for pack in packs:
            dataset = model.DecisionDataset(pack, store, policy.processor)
            batch = collate([dataset[i] for i in range(len(pack))])
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            logits = policy.forward_batch(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                mm_token_type_ids=batch['mm_token_type_ids'], image_grid_thw=batch['image_grid_thw'],
                pixel_values=batch['pixel_values'], exec_index=batch['exec_index'],
                action_index=batch['action_index'])
            ce = torch.nn.functional.cross_entropy(logits, batch['targets'], reduction='none')
            ce_sum += float(ce.sum())
            n += len(pack)
            for t_, p_ in zip(batch['targets'].tolist(), logits.argmax(-1).tolist()):
                confusion[t_][p_] += 1
    support = [sum(r) for r in confusion]
    total = max(1, sum(support))
    recall = [confusion[i][i] / support[i] if support[i] else None for i in range(4)]
    macro = sum(r for r in recall if r is not None) / 4
    accuracy = sum(confusion[i][i] for i in range(4)) / total
    predictions = [sum(confusion[t][p] for t in range(4)) for p in range(4)]
    forward_baseline = support[0] / total  # always-predict-F accuracy on this panel
    result = dict(status='PANEL_EVAL_COMPLETED', unix=time.time(),
                  checkpoint=str(args.checkpoint), kernel=spec['kernel'], panel=spec['panel_sha256'],
                  ce=ce_sum / n, accuracy=accuracy, action_recall=recall, macro_recall=macro,
                  target_distribution=support, prediction_distribution=predictions,
                  always_forward_baseline=forward_baseline,
                  accuracy_minus_always_forward=accuracy - forward_baseline,
                  confusion=confusion, n=n,
                  note='teacher-forced panel diagnostics; not DEV, not closed-loop navigation')
    args.out.mkdir(parents=True, exist_ok=True)
    name = 'EVAL_%s.json' % Path(args.checkpoint).stem
    with (args.out / name).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(ce=round(result['ce'], 4), macro_recall=round(macro, 4),
                          recall=[round(r, 4) if r is not None else None for r in recall],
                          delta_vs_always_forward=round(result['accuracy_minus_always_forward'], 4))))


if __name__ == '__main__':
    main()
