"""Read-only fixed V11 actors on the same precompiled real action forks."""
import os
from pathlib import Path
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    c = train.c
    began = time.monotonic()
    data = c.read(HERE.parent/'multifamily_v7/DATA.json')
    cache_path = HERE.parent/'multifamily_v7/run_001/FEATURES.pt'
    assert c.sha(cache_path) == c.read(cache_path.parent/'FEATURE_RESULT.json')['file_sha256']
    cache = {k: v.float() for k, v in torch.load(cache_path, map_location='cpu', weights_only=True).items()}
    prior = HERE.parent/'query_semantics_v11/run_001'
    admission = c.read(HERE/'ACTOR_ADMISSION.json')
    rows = {}
    for seed in (1209, 1210, 1211):
        for arm in ('B2', 'Ours'):
            key = f'{arm}_{seed}'
            path = prior/f'{key}_MEMORY.pt'
            net = train.models.MemoryPolicy(2048, len(data['query_vocabulary']), 8, 64, .99)
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            net.no_memory = False
            identity = c.model_identity(net)['sha256']
            assert identity == c.read(prior/f'{key}_RESULT.json')['final_state_sha256']
            measured = train.branch_metrics.evaluate(net, cache, data, admission)
            assert c.model_identity(net)['sha256'] == identity
            rows[key] = dict(branch_actions=measured, checkpoint_sha256=c.sha(path), state_sha256=identity)
    c.write(HERE/'PRIOR_TEACHER_BRANCH_EVALUATION.json', dict(runs=rows, cpu_seconds=time.monotonic()-began,
        optimizer_updates=0, new_qwen_forwards=0, gpu_hours=0,
        actor_admission_sha256=c.sha(HERE/'ACTOR_ADMISSION.json'),
        scope='Post-hoc read-only diagnostic of all six final V11 actors on the precompiled V13 fork set. No model/seed selection or closed-loop SR.'), True)
    print({k: v['branch_actions']['summaries'] for k, v in rows.items()})


if __name__ == '__main__':
    main()
