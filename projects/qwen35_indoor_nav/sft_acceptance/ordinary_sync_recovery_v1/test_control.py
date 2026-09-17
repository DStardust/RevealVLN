"""CPU-only regression tests, including actual three-rank skewed collectives."""
import datetime
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control
import data
import torch


def unit_tests():
    n = 0
    # The old predicate disagrees under normal rank clock drift.
    assert [v - 100 >= 10 for v in (109.99, 110.01, 110.00)] == [False, True, True]
    for step in (30400, 30401, 30499, 30500):
        assert len({control.report_due(step, 20) for _ in (109.99, 110.01, 999)}) == 1
    n += 1
    for invalid in (0, -1, None):
        try:
            control.report_due(1, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('INVALID_CADENCE_ACCEPTED')
    n += 1
    assert control.report_due(7, 20, True)
    assert control.report_due(7, 20, False, 1)
    n += 1
    assert control.stop_mask([], 10, 20, 30, 40, 50, 60) == 0
    assert control.stop_mask(['SIGTERM'], 20, 20, 40, 40, 60, 60) == 15
    assert control.stop_reasons(8) == ['BUDGET:max_decisions']
    n += 1
    samples = [dict(est=(i * 37) % 500 + 50) for i in range(700)]
    plans = [data.plan_epoch_batches(samples, 1024, 1109, e, 3) for e in range(3)]
    cursor = dict(epoch=1, position=3, updates=len(plans[0][0]) + 3, decisions=99999)
    counts = [control.processed_by_rank(plans, cursor, r) for r in range(3)]
    assert counts == [sum(map(len, plans[0][r])) + sum(map(len, plans[1][r][:3])) for r in range(3)]
    assert sum(counts) != 99999 * 3
    n += 1
    assert control.charged_decisions(2236987, 900) == 2237887
    # Replaying 900 decisions after rollback is charged again, not erased.
    assert control.charged_decisions(2237887, 900) == 2238787
    n += 1
    # Model changes are device placement only; algorithm and all tensor shapes
    # remain the frozen model (strip the explicit placement amendment).
    source = (HERE / 'model.py').read_text()
    assert "device='cuda:0'" not in source and ".to('cuda:0')" not in source
    n += 1
    return n


def distributed_test():
    rank = int(os.environ['RANK'])
    torch.distributed.init_process_group('gloo', timeout=datetime.timedelta(seconds=20))
    events = []
    for step in range(1, 31):
        time.sleep(.002 * rank)
        stopped = control.agree_stop(torch, 1 if rank == 1 and step == 17 else 0, 'cpu', 3)
        due = control.report_due(step, 5, False, stopped)
        if due:
            packed = torch.tensor([rank + 1.] * 4, dtype=torch.float64)
            torch.distributed.all_reduce(packed)
            assert packed.tolist() == [6.] * 4
            confusion = torch.ones((4, 4), dtype=torch.float64)
            torch.distributed.all_reduce(confusion)
            assert confusion.sum().item() == 48
            events.append(step)
        if stopped:
            break
    assert events == [5, 10, 15, 17], (rank, events)
    # A rank-local wall-time expiry propagates to all ranks without divergent IO.
    mask = control.stop_mask([], 21 if rank == 2 else 19, 20, 17, 100, 0, 100)
    assert control.agree_stop(torch, mask, 'cpu', 3) == 2
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(json.dumps(dict(rank=rank, status='PASS', events=events, tests='REAL_GLOO_SKEW_SINGLE_RANK_STOP_WALL_BUDGET')), flush=True)


if __name__ == '__main__':
    if '--distributed' in sys.argv:
        distributed_test()
    else:
        print(json.dumps(dict(status='PASS', unit_tests=unit_tests())))
