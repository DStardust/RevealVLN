"""CPU read-only teacher-action probes; none is a new navigation policy.

Compare learned recurrent state with zero state and a FIT-derived constant state.
Neither intervention is a matched semantic donor, so these are sensitivity probes,
not proofs that the policy uses a particular task-history variable.
"""
from pathlib import Path
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train
c = train.c


def scores(logits, target, native):
    raw = logits.argmax(-1)
    prediction = torch.where(native.argmax(-1) == 3, 3, raw)
    matrix = torch.bincount(target * 4 + prediction, minlength=16).reshape(4, 4)
    return dict(correct=int(matrix.diag().sum()), decisions=len(target),
                accuracy=float(matrix.diag().sum() / len(target)),
                matrix_true_rows_predicted_columns=matrix.tolist(),
                raw_head_correct=int((raw == target).sum()))


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    began = time.monotonic()
    natural = HERE.parent / 'natural_transfer_v9'
    identity = c.read(natural / 'run_001/FEATURE_RESULT.json')
    cache_path = natural / 'run_001/FEATURES.pt'
    assert c.sha(cache_path) == identity['file_sha256']
    cache = {k: v.float() for k, v in torch.load(cache_path, map_location='cpu', weights_only=True).items()}
    data = c.read(natural / 'DATA.json')
    special = c.read(HERE.parent / 'multifamily_v7/DATA.json')
    result = c.read(HERE / 'run_001/RESULT.json')
    config = c.read(HERE / 'PROTOCOL.json')
    records = data['records']
    rows = {}
    for seed in config['seeds']:
        for arm in config['arms']:
            key = f'{arm}_{seed}'
            path = HERE / 'run_001' / f'{key}_MEMORY.pt'
            net = train.models.MemoryPolicy(2048, len(special['query_vocabulary']), 8, 64, .99,
                                           no_memory=arm == 'N0').eval()
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            expected = result['runs'][key]['final_state_sha256']
            assert c.model_identity(net)['sha256'] == expected
            fit_means, check = [], []
            for start in range(0, len(records), 8):
                selected = records[start:start+8]
                batch = train.ordinary_batch(selected, 'cpu')
                features = cache['features'][batch['indices']]
                states, _ = net.encode(features)
                for i, record in enumerate(selected):
                    n = len(record['features'])
                    memory = states[i, :n]
                    if record['partition'] == 'fit':
                        fit_means.append(memory.mean(0))
                    else:
                        check.append((memory, features[i, :n],
                                      cache['logits'][batch['indices'][i, :n]], batch['targets'][i, :n]))
                c.write(HERE / 'MEMORY_DIAGNOSIS_PROGRESS.json',
                        dict(unix=time.time(), key=key, routes=min(start+8, len(records)), total_routes=len(records)))
            assert len(fit_means) == 508 and len(check) == 60
            constant = torch.stack(fit_means).mean(0)
            outputs = {name: [] for name in ('full', 'zero', 'fit_constant')}
            native_rows, targets = [], []
            for memory, features, native, target in check:
                for name, value in (('full', memory), ('zero', torch.zeros_like(memory)),
                                    ('fit_constant', constant.unsqueeze(0).expand_as(memory))):
                    output = net.action_logits(value, native, features)
                    assert bool(torch.isfinite(output).all())
                    outputs[name].append(output)
                native_rows.append(native)
                targets.append(target)
            outputs = {name: torch.cat(value) for name, value in outputs.items()}
            native, target = torch.cat(native_rows), torch.cat(targets)
            assert len(target) == 3769
            protected = {name: torch.where(native.argmax(-1) == 3, 3, value.argmax(-1))
                         for name, value in outputs.items()}
            probes = {name: dict(**scores(value, target, native),
                                max_logit_delta_from_full=float((value-outputs['full']).abs().max()),
                                protected_action_changes_from_full=int((protected[name] != protected['full']).sum()))
                      for name, value in outputs.items()}
            if arm == 'N0':
                assert torch.equal(outputs['full'], outputs['zero'])
                assert torch.equal(outputs['full'], outputs['fit_constant'])
            recorded_correct = sum(row['correct'] for row in result['runs'][key]['ordinary']['rows']
                                   if row['partition'] == 'check')
            assert c.model_identity(net)['sha256'] == expected
            rows[key] = dict(probes=probes, native=scores(native, target, native),
                             recorded_gpu_raw_head_correct=recorded_correct,
                             cpu_minus_recorded_gpu_correct=probes['full']['raw_head_correct']-recorded_correct,
                             checkpoint_sha256=c.sha(path), state_sha256=expected, parameters_unchanged=True)
            print(key, {name: value['accuracy'] for name, value in probes.items()}, flush=True)
    c.write(HERE / 'MEMORY_DIAGNOSIS.json', dict(status='READ_ONLY_TEACHER_MEMORY_SENSITIVITY',
        runs=rows, source_sha256=c.sha(Path(__file__)), feature_sha256=identity['file_sha256'],
        data_sha256=c.sha(natural / 'DATA.json'), gpu_used=False, optimizer_updates=0,
        new_navigation_episodes=0, seconds=time.monotonic()-began,
        constant_definition='Equal mean over508 FIT trajectory mean memories; no labels used to compute constant',
        scope='60 exposed CHECK teacher routes,3769 decisions; native STOP preserved. No closed-loop claim.',
        limitation='Zero and constant states may be out of distribution; neither is a certified task-state-matched sham. Sensitivity cannot identify a unique semantic history variable.',
        actions=['move_forward', 'turn_left', 'turn_right', 'STOP']), True)


if __name__ == '__main__':
    main()
