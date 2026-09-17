"""One-batch memory diagnostic for the v3 fallback path. No training state."""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mib():
    import torch
    return dict(alloc=round(torch.cuda.memory_allocated() / 2**20),
                reserved=round(torch.cuda.memory_reserved() / 2**20))


def main():
    max_tokens = int(sys.argv[1])
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    torch.use_deterministic_algorithms(True)
    data = load_module('d', HERE / 'data.py')
    model = load_module('m', HERE / 'model.py')
    acc = json.loads((HERE / 'acceptance/ACCEPTANCE.json').read_text())
    rows, report = data.load_rows()
    samples = data.load_sample_index(HERE / 'SAMPLE_INDEX.jsonl', acc['sample_index_sha256'],
                                     sum(r['decisions'] for r in rows))
    marks = {}
    marks['before_model'] = mib()
    policy = model.build_policy(1109)
    policy.train()
    marks['after_model'] = mib()
    store = data.SampleStore(rows)
    plan = data.plan_epoch_batches(samples, max_tokens, 1109, 0, 1)[0]
    batch_ids = plan[0]
    print(json.dumps(dict(batch_size=len(batch_ids),
                          est_sum=sum(samples[i]['est'] for i in batch_ids))), flush=True)
    dataset = model.DecisionDataset(samples, store, policy.processor)
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id, policy.exec_sid,
                                 policy.query_sid, policy.base.config.image_token_id)
    batch = collate([dataset[i] for i in batch_ids])
    marks['after_collate_cpu'] = mib()
    batch = {k: (v.cuda() if hasattr(v, 'to') else v) for k, v in batch.items()}
    marks['after_to_gpu'] = mib()
    print(json.dumps(dict(padded_shape=list(batch['input_ids'].shape))), flush=True)
    logits = policy.forward_batch(**{k: batch[k] for k in
                                     ('input_ids', 'attention_mask', 'mm_token_type_ids',
                                      'image_grid_thw', 'pixel_values', 'exec_index',
                                      'action_index')})
    marks['after_forward'] = mib()
    ce = torch.nn.functional.cross_entropy(logits, batch['targets'], reduction='none')
    loss = (ce * batch['weights']).sum() / batch['weights'].sum()
    loss.backward()
    marks['after_backward'] = mib()
    print(json.dumps(marks), flush=True)
    summary = torch.cuda.memory_summary().splitlines()
    for line in summary[:22]:
        print(line)


if __name__ == '__main__':
    main()
