"""Rank-consistent control and explicit progress-versus-compute accounting."""


def report_due(updates, every, end_of_epoch=False, agreed_stop=False):
    if not isinstance(every, int) or every <= 0:
        raise ValueError('POSITIVE_UPDATE_LOG_INTERVAL_REQUIRED')
    return updates % every == 0 or end_of_epoch or bool(agreed_stop)


def stop_mask(signals, unix, deadline, updates, max_updates, charged, max_decisions):
    return (int(bool(signals)) | (2 if unix >= deadline else 0) |
            (4 if updates >= max_updates else 0) | (8 if charged >= max_decisions else 0))


def agree_stop(torch, mask, device, world):
    # A fixed int64 vector, fixed collective, unconditional call on every rank.
    flags = torch.tensor([int(bool(mask & (1 << i))) for i in range(4)],
                         dtype=torch.int64, device=device)
    if world > 1:
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
    return sum(int(v) << i for i, v in enumerate(flags.tolist()))


def stop_reasons(mask):
    return [name for i, name in enumerate(('SIGNAL', 'BUDGET:wall_seconds',
                                         'BUDGET:max_updates', 'BUDGET:max_decisions'))
            if mask & (1 << i)]


def processed_by_rank(plans, cursor, rank):
    epoch, position = cursor['epoch'], cursor['position']
    if not 0 <= epoch <= len(plans):
        raise ValueError('BAD_EPOCH')
    if epoch == len(plans):
        if position != 0:
            raise ValueError('COMPLETED_CURSOR_POSITION')
    elif not 0 <= position <= len(plans[epoch][rank]):
        raise ValueError('BAD_POSITION')
    expected_updates = sum(len(p[rank]) for p in plans[:epoch]) + position
    if expected_updates != cursor['updates']:
        raise ValueError('CURSOR_UPDATE_PLAN_MISMATCH')
    return sum(len(b) for p in plans[:epoch] for b in p[rank]) + (
        sum(len(b) for b in plans[epoch][rank][:position]) if epoch < len(plans) else 0)


def charged_decisions(base, segment_decisions):
    if base < 0 or segment_decisions < 0:
        raise ValueError('NEGATIVE_COMPUTE')
    return int(base + segment_decisions)
