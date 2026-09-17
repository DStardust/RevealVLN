"""CPU-only comparison of reference vs candidate kernel probe dumps.

Applies the pre-registered PROTOCOL.json thresholds. No GPU, no network,
no mutation outside this version directory. FAIL stays FAIL.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(ref, cand, thresholds, counters_cap):
    import torch
    gates = {}
    ref_logits, cand_logits = ref['logits'], cand['logits']
    same_keys = set(ref_logits) == set(cand_logits)
    gates['same_decision_keys'] = same_keys
    if same_keys:
        diffs = [float((ref_logits[k] - cand_logits[k]).abs().max()) for k in ref_logits]
        agree = all(int(ref_logits[k].argmax(-1).item()) == int(cand_logits[k].argmax(-1).item())
                    for k in ref_logits)
        gates['logits_max_abs_diff'] = max(diffs)
        gates['logits_max_abs_diff_pass'] = max(diffs) <= thresholds['logits_max_abs_diff']
        gates['argmax_agreement'] = bool(agree)
        gates['argmax_agreement_pass'] = bool(agree)
    same_mem = set(ref['memories']) == set(cand['memories'])
    gates['same_memory_keys'] = same_mem
    if same_mem:
        mdiff = max(float((ref['memories'][k] - cand['memories'][k]).abs().max()) for k in ref['memories'])
        gates['final_memory_max_abs_diff'] = mdiff
        gates['final_memory_max_abs_diff_pass'] = mdiff <= thresholds['final_memory_max_abs_diff']
    loss_rel = abs(cand['ce_sum'] - ref['ce_sum']) / max(abs(ref['ce_sum']), 1e-12)
    gates['loss_relative_diff'] = loss_rel
    gates['loss_relative_diff_pass'] = loss_rel <= thresholds['loss_relative_diff']

    same_trainable = ref['trainable'] == cand['trainable']
    gates['same_trainable_names'] = same_trainable
    grad_worst = {}
    update_worst = {}
    if same_trainable:
        for name in ref['trainable']:
            r, c = ref['grads'][name].float(), cand['grads'][name].float()
            rel = float((r - c).norm() / r.norm().clamp_min(1e-12))
            # Zero-norm gradients (e.g. fresh-LoRA A before the first update) have
            # no meaningful direction; relative L2 alone gates them.
            if max(float(r.norm()), float(c.norm())) > 1e-12:
                cos = float(torch.nn.functional.cosine_similarity(r.flatten(), c.flatten(), dim=0))
            else:
                cos = None
            grad_worst[name] = dict(relative_l2=rel, cosine=cos)
            ru, cu = ref['update_deltas'][name].float(), cand['update_deltas'][name].float()
            update_worst[name] = float((ru - cu).norm() / ru.norm().clamp_min(1e-12))
        worst_grad_rel = max(v['relative_l2'] for v in grad_worst.values())
        cosines = [v['cosine'] for v in grad_worst.values() if v['cosine'] is not None]
        worst_grad_cos = min(cosines) if cosines else None
        worst_update_rel = max(update_worst.values())
        gates['grad_relative_l2_max'] = worst_grad_rel
        gates['grad_cosine_min'] = worst_grad_cos
        gates['grad_pass'] = (worst_grad_rel <= thresholds['grad_relative_l2']
                              and (worst_grad_cos is None or worst_grad_cos >= thresholds['grad_cosine_min']))
        gates['update_relative_l2_max'] = worst_update_rel
        gates['update_pass'] = worst_update_rel <= thresholds['update_relative_l2']

    total_decisions = ref['counters']['decisions'] + cand['counters']['decisions']
    total_updates = ref['counters']['updates'] + cand['counters']['updates']
    gates['total_forward_decisions'] = total_decisions
    gates['total_optimizer_updates'] = total_updates
    gates['decision_cap_pass'] = total_decisions <= counters_cap['max_forward_decisions']
    gates['update_cap_pass'] = total_updates <= counters_cap['max_probe_optimizer_updates']
    gates['reference_fla_absent'] = not ref['fla_active']
    gates['candidate_fla_active'] = bool(cand['fla_active'])
    gates['determinism_flags'] = [bool(ref['deterministic_algorithms']), bool(cand['deterministic_algorithms'])]
    gates['same_preflight_binding'] = ref['preflight_binding'] == cand['preflight_binding']
    gates['same_decision_set'] = ([(c['chunk'], c['record_id'], c['steps']) for c in ref['decision_set']]
                                  == [(c['chunk'], c['record_id'], c['steps']) for c in cand['decision_set']])
    gates['candidate_memory_bytes'] = cand['peak_reserved_bytes']
    gates['memory_cap_pass'] = (cand['peak_reserved_bytes'] <= counters_cap['max_own_gpu_memory_bytes']
                                and ref['peak_reserved_bytes'] <= counters_cap['max_own_gpu_memory_bytes'])

    ref_tps = next(t['decisions_per_second'] for t in ref['timing'] if t['phase'] == 'measured')
    cand_tps = next(t['decisions_per_second'] for t in cand['timing'] if t['phase'] == 'measured')
    gates['timing'] = dict(reference_decisions_per_second=ref_tps, candidate_decisions_per_second=cand_tps,
                           kernel_only_speedup=cand_tps / ref_tps if ref_tps else None,
                           scope='kernel swap only, batch=1, short record; not the 10x end-to-end gate')

    pass_keys = [k for k in gates if k.endswith('_pass')] + ['argmax_agreement', 'same_decision_keys',
        'same_memory_keys', 'same_trainable_names', 'reference_fla_absent', 'candidate_fla_active',
        'same_preflight_binding', 'same_decision_set']
    gates['status'] = 'PASS' if all(gates.get(k) for k in pass_keys) else 'FAIL'
    gates['grad_detail'] = grad_worst
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    import torch
    # Self-produced dumps; TorchVersion is recorded for environment evidence.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ref = torch.load(args.run / 'PROBE_REFERENCE.pt', map_location='cpu', weights_only=True)
        cand = torch.load(args.run / 'PROBE_CANDIDATE.pt', map_location='cpu', weights_only=True)
    gates = evaluate(ref, cand, protocol['thresholds'], protocol['budget'])
    result = dict(unix=time.time(), run=str(args.run),
                  reference_dump_sha256=sha256(args.run / 'PROBE_REFERENCE.pt'),
                  candidate_dump_sha256=sha256(args.run / 'PROBE_CANDIDATE.pt'),
                  thresholds=protocol['thresholds'], gates=gates,
                  fail_preserved=True, no_threshold_relaxed=True)
    with (args.run / 'COMPARISON.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
    print(json.dumps({k: v for k, v in gates.items() if k != 'grad_detail'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
