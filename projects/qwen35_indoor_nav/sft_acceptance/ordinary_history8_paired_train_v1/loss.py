"""One global weighted CE across ranks and accumulated microbatches."""
def local_micro_loss(logits, targets, weights, global_weight_sum, world_size):
    import torch
    if world_size < 1 or global_weight_sum <= 0:
        raise ValueError('GLOBAL_WEIGHT_NORMALIZATION')
    if logits.ndim != 2 or logits.shape[0] != targets.numel() or targets.shape != weights.shape:
        raise ValueError('MICROBATCH_SHAPE')
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError('INVALID_WEIGHTS')
    losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    # Existing sync_grads averages ranks; this exactly cancels that world factor.
    # No per-microbatch normalization: unequal weights must remain global.
    return (losses * weights).sum() * (world_size / global_weight_sum)

