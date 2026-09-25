"""One causal choice of the remaining frozen policy, never an outcome lookup."""
import torch


def forced_action(prefix, environment_step, generation_offset, eos_token):
    if generation_offset is None or environment_step >= len(prefix):
        return None
    assert environment_step % 4 == 0 and len(prefix) % 4 == 0
    return prefix[environment_step + generation_offset] if generation_offset < 4 else eos_token


def choose_once(gate, actor, native, proposed, token_ids, accepted):
    if accepted is None and not torch.equal(native.argmax(-1), proposed.argmax(-1)):
        logits = gate(actor, native[:, token_ids].float(), proposed[:, token_ids].float())
        if not torch.isfinite(logits).all():raise ValueError('NONFINITE_GATE_LOGITS')
        probability = logits.softmax(-1)
        accepted = bool((probability[:, 1] > 2 * probability[:, 2]).item())
        evidence = dict(probability=probability[0].tolist(), accepted=accepted)
    else:
        evidence = None
    return (proposed if accepted is True else native), accepted, evidence


def disagreement(record):
    return not record.get('forced_prefix_query', False) and record['native_token'] != record['method_token']
