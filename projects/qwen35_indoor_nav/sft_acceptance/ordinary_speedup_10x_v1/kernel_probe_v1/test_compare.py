"""CPU tests for compare.evaluate; synthetic tensors only, no GPU/model/data."""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import compare

THRESHOLDS = dict(logits_max_abs_diff=0.05, final_memory_max_abs_diff=0.02,
                  loss_relative_diff=1e-3, grad_relative_l2=0.05, grad_cosine_min=0.9999,
                  update_relative_l2=0.15)
CAPS = dict(max_forward_decisions=512, max_probe_optimizer_updates=8,
            max_own_gpu_memory_bytes=30064771072)


def make_dump(seed, scale=0.0):
    g = torch.Generator().manual_seed(seed)
    logits = {'short:%d' % t: torch.randn(1, 4, generator=g) * 5 for t in range(4)}
    logits['longest_stop:9'] = torch.randn(1, 4, generator=g) * 5
    memories = {'short': torch.randn(8, 16, generator=g), 'longest_stop': torch.randn(8, 16, generator=g)}
    grads = {'a': torch.randn(8, 8, generator=g), 'b': torch.randn(4, 4, generator=g)}
    deltas = {'a': torch.randn(8, 8, generator=g) * 1e-4, 'b': torch.randn(4, 4, generator=g) * 1e-4}
    dump = dict(logits=logits, memories=memories, ce_sum=3.5, trainable=['a', 'b'],
                grads=grads, update_deltas=deltas,
                counters=dict(decisions=49, updates=1), fla_active=False,
                deterministic_algorithms=True,
                preflight_binding=dict(a=1), decision_set=[dict(chunk='short', record_id='x', steps=[0, 1])],
                peak_reserved_bytes=1024,
                timing=[dict(phase='measured', decisions=16, seconds=8.0, decisions_per_second=2.0)])
    if scale:
        for key in dump['logits']:
            dump['logits'][key] = dump['logits'][key] + scale
        for key in dump['memories']:
            dump['memories'][key] = dump['memories'][key] + scale * 0.1
        dump['grads'] = {k: v * (1 + scale * 0.01) for k, v in dump['grads'].items()}
        dump['update_deltas'] = {k: v * (1 + scale * 0.01) for k, v in dump['update_deltas'].items()}
        dump['ce_sum'] += scale * 0.0001
        dump['fla_active'] = True
    return dump


def test_identical_passes():
    ref = make_dump(0)
    cand = make_dump(0)
    cand['fla_active'] = True
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'PASS', gates
    assert gates['kernel_only_speedup'] if 'kernel_only_speedup' in gates else True


def test_logit_diff_fails():
    ref = make_dump(0)
    cand = make_dump(1, scale=1.0)
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'FAIL'
    assert not gates['logits_max_abs_diff_pass']


def test_argmax_mismatch_fails():
    ref = make_dump(0)
    cand = make_dump(0)
    cand['fla_active'] = True
    key = sorted(cand['logits'])[0]
    cand['logits'][key] = cand['logits'][key].flip(-1)
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'FAIL'
    assert not gates['argmax_agreement']


def test_reference_fla_leak_fails():
    ref = make_dump(0)
    ref['fla_active'] = True
    cand = make_dump(0)
    cand['fla_active'] = True
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'FAIL'


def test_grad_divergence_fails():
    ref = make_dump(0)
    cand = make_dump(0)
    cand['fla_active'] = True
    cand['grads']['a'] = -cand['grads']['a']
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'FAIL'
    assert not gates['grad_pass']


def test_decision_cap_fails():
    ref = make_dump(0)
    cand = make_dump(0)
    cand['fla_active'] = True
    cand['counters']['decisions'] = 500
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'FAIL'
    assert not gates['decision_cap_pass']


def test_zero_gradient_tensors_pass():
    ref = make_dump(0)
    cand = make_dump(0)
    cand['fla_active'] = True
    for dump in (ref, cand):
        dump['grads']['b'] = torch.zeros(4, 4)
        dump['update_deltas']['b'] = torch.zeros(4, 4)
    gates = compare.evaluate(ref, cand, THRESHOLDS, CAPS)
    assert gates['status'] == 'PASS', gates
    assert gates['grad_detail']['b']['cosine'] is None


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
