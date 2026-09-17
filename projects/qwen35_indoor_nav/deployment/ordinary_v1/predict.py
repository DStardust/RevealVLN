"""Batch-one offline inference with the retained ordinary navigation baseline."""
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(row):
    if set(row) != {'instruction', 'images', 'executed'}:
        raise ValueError('Expected only instruction, images, executed')
    if not isinstance(row['instruction'], str) or not row['instruction'].strip():
        raise ValueError('Instruction must be a nonempty string')
    if not isinstance(row['images'], list) or not 1 <= len(row['images']) <= 2:
        raise ValueError('Supply one or two chronological RGB image paths')
    if not all(isinstance(path, str) for path in row['images']):
        raise ValueError('Image paths must be strings')
    if not isinstance(row['executed'], list) or len(row['executed']) > 8:
        raise ValueError('Supply at most eight actually executed actions')
    if any(a not in ('move_forward', 'turn_left', 'turn_right') for a in row['executed']):
        raise ValueError('Executed history accepts move_forward, turn_left, turn_right')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSONL; image paths relative to this file')
    parser.add_argument('--output', type=Path, required=True, help='New output JSONL, never overwritten')
    parser.add_argument('--gpu', type=int, required=True, help='Physical GPU index allocated by the caller')
    args = parser.parse_args()
    rows = [validate(json.loads(line)) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or args.output.exists():
        raise ValueError('Nonempty input and a new output path are required')
    card = json.loads((HERE / 'MODEL_CARD.json').read_text())
    checkpoint = LINE / card['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != card['checkpoint_sha256']:
        raise ValueError('Checkpoint identity mismatch')
    model_path = LINE / 'sft_acceptance/ordinary_sync_recovery_v1/model.py'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != card['model_source_sha256']:
        raise ValueError('Model implementation identity mismatch')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'CONDA_PREFIX'):
        os.environ.pop(key, None)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key, name in dict(HF_HOME='hf', XDG_CACHE_HOME='xdg', TORCH_HOME='torch', TRITON_CACHE_DIR='triton',
                          CUDA_CACHE_PATH='cuda', TMPDIR='tmp').items():
        folder = HERE / 'cache' / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    speed = LINE / 'sft_acceptance/ordinary_speedup_10x_v1'
    for path in ('official_einops_0_8_1/deps', 'official_fla_0_5_2/deps'):
        sys.path.insert(0, str(speed / path))
    import fla.ops.gated_delta_rule
    import torch
    from PIL import Image
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(25 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (x.cell_contents for x in bound.__closure__)))
    if not getattr(cells.get('implementation'), '__module__', '').startswith('fla.'):
        raise RuntimeError('Accepted official FLA implementation is not active')
    model = load_module('ordinary_policy', model_path)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['cursor']['updates'] != card['checkpoint_updates']:
        raise ValueError('Checkpoint update binding mismatch')
    policy = model.build_policy(card['seed'])
    model.load_trainable(policy, state['trainable'])
    del state
    policy.eval()
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id, policy.exec_sid,
                                policy.query_sid, policy.base.config.image_token_id)

    class Store:
        def get(self, idx, t):
            row = rows[idx]
            images = []
            for name in row['images']:
                with Image.open(args.input.parent / name) as image:
                    if image.size != (224, 224):
                        raise ValueError('This baseline expects the original 224 by 224 camera RGB')
                    images.append(image.convert('RGB'))
            return dict(row, images=images)

    dataset = model.DecisionDataset([dict(record_idx=i, t=0, target=0, weight=1.) for i in range(len(rows))],
                                    Store(), policy.processor)
    with args.output.open('x') as output, torch.inference_mode():
        for index in range(len(rows)):
            started = time.perf_counter()
            batch = collate([dataset[index]])
            batch.pop('targets')
            batch.pop('weights')
            logits = policy.forward_batch(**{k: v.to('cuda:0') for k, v in batch.items()})
            torch.cuda.synchronize()
            values = logits[0].float().cpu().tolist()
            record = dict(index=index, action=model.ACTIONS[max(range(4), key=values.__getitem__)],
                          logits=values, preprocessing_and_forward_seconds=time.perf_counter() - started,
                          checkpoint_sha256=card['checkpoint_sha256'])
            output.write(json.dumps(record, allow_nan=False) + '\n')
            output.flush()


if __name__ == '__main__':
    main()
