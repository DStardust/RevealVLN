"""Unit-level numerical isolation: transformers torch fallback vs official FLA op.

One process, no model load, no data, no training. Captures the production
fallback function FIRST (fla absent at modeling import), then imports the
official FLA op, then compares forward outputs and input gradients on fixed
synthetic tensors at the production GDN shapes (B=1, H=HV=16, K=V=128).
Diagnostic only; the end-to-end gate stays kernel_probe_v1/run_v1/COMPARISON.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SPEED = HERE.parent
FLA_DEPS = SPEED / 'official_fla_0_5_2/deps'
EINOPS_DEPS = SPEED / 'official_einops_0_8_1/deps'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    out_dir = HERE / 'run_unit_v1'
    out_dir.mkdir(exist_ok=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'), 'EXACT_GPU_UUID_REQUIRED')
    import torch
    torch.use_deterministic_algorithms(True)

    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    fallback = modeling.torch_chunk_gated_delta_rule
    require(fallback.__closure__, 'FALLBACK_CLOSURE')
    cells = dict(zip(fallback.__code__.co_freevars, (c.cell_contents for c in fallback.__closure__)))
    fallback_impl = cells.get('implementation')
    require(getattr(fallback_impl, '__module__', '').endswith('modeling_qwen3_5'),
            'FALLBACK_NOT_CAPTURED:' + str(getattr(fallback_impl, '__module__', None)))
    require('fla' not in sys.modules, 'FLA_PREMATURELY_PRESENT')

    sys.path.insert(0, str(EINOPS_DEPS))
    sys.path.insert(0, str(FLA_DEPS))
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_fn

    torch.manual_seed(1109)
    dev = 'cuda:0'
    results = {}
    for tag, T in (('t320', 320), ('t805', 805)):
        B, H, HV, K, V = 1, 16, 16, 128, 128
        base = dict(
            q=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev) * 0.5,
            k=torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev) * 0.5,
            v=torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev),
            g=-torch.rand(B, T, HV, dtype=torch.float32, device=dev) * 0.5,
            beta=torch.rand(B, T, HV, dtype=torch.bfloat16, device=dev),
        )
        dout = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)

        def run(fn):
            tensors = {name: value.detach().clone().requires_grad_(True)
                       for name, value in base.items()}
            out, last = fn(tensors['q'], tensors['k'], tensors['v'], g=tensors['g'],
                           beta=tensors['beta'], initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=True)
            require(last is None, 'UNEXPECTED_FINAL_STATE')
            out.backward(dout)
            return out.detach(), {name: tensors[name].grad.detach().float().cpu()
                                  for name in tensors}

        ref_out, ref_grads = run(fallback)
        cand_out, cand_grads = run(fla_fn)
        out_diff = float((ref_out.float() - cand_out.float()).abs().max())
        grad_stats = {}
        for name in ref_grads:
            r, c = ref_grads[name], cand_grads[name]
            rel = float((r - c).norm() / r.norm().clamp_min(1e-12))
            cos = float(torch.nn.functional.cosine_similarity(r.flatten(), c.flatten(), dim=0))
            grad_stats[name] = dict(relative_l2=rel, cosine=cos,
                                    ref_norm=float(r.norm()), cand_norm=float(c.norm()))
        results[tag] = dict(T=T, out_max_abs_diff=out_diff,
                            out_ref_norm=float(ref_out.float().norm()),
                            grad=grad_stats)
        print(json.dumps({tag: results[tag]}, allow_nan=False), flush=True)

    path = out_dir / 'UNIT_KERNEL_CHECK.json'
    require(not path.exists(), 'UNIT_RESULT_EXISTS')
    with path.open('x') as stream:
        json.dump(dict(unix=time.time(), results=results,
                       note='diagnostic only; end-to-end gate is COMPARISON.json'), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
