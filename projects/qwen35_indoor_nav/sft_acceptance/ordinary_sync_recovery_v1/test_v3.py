"""CPU tests for ordinary baseline v3: no GPU, no model weights, no snapshot reads."""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import data
import model
import train


def test_inflection_weight():
    actions = ['move_forward', 'move_forward', 'turn_left', 'move_forward', 'STOP']
    got = [data.inflection_weight(actions, t, 3.2) for t in range(5)]
    assert got == [3.2, 1.0, 3.2, 3.2, 3.2], got


def test_plan_epoch_batches_deterministic():
    samples = [dict(est=e) for e in [100, 300, 200, 400, 250, 150, 350, 320, 280, 310, 290, 270]]
    p1 = data.plan_epoch_batches(samples, 512, 1109, 0, 1)
    p2 = data.plan_epoch_batches(samples, 512, 1109, 0, 1)
    assert p1 == p2, 'nondeterministic plan'
    p3 = data.plan_epoch_batches(samples, 512, 1109, 1, 1)
    assert p1 != p3, 'epoch does not change plan'
    # batches are token-budget packed from a length-sorted order
    for batch in p1[0]:
        ests = sorted(samples[i]['est'] for i in batch)
        assert ests == sorted(ests)
        assert sum(samples[i]['est'] for i in batch) <= 512 + max(ests)  # greedy pack bound


def test_plan_epoch_batches_sharding():
    samples = [dict(est=(i * 37) % 500 + 50) for i in range(1000)]
    world = 4
    plans = data.plan_epoch_batches(samples, 1024, 1109, 0, world)
    assert len(plans) == world
    assert len({len(x) for x in plans}) == 1, 'uneven shards'
    seen = []
    for shard in plans:
        for batch in shard:
            seen.extend(batch)
    all_batches = data.plan_epoch_batches(samples, 1024, 1109, 0, 1)[0]
    usable = len(all_batches) - len(all_batches) % world
    expected = [i for b in all_batches[:usable] for i in b]
    assert sorted(seen) == sorted(expected), 'shard union mismatch'
    assert len(seen) == len(set(seen)), 'duplicate samples across shards'


def test_collate_assembly():
    pad_id, query_sid = 99, 900
    exec_sid = {'move_forward': 901, 'turn_left': 902, 'turn_right': 903}
    image_token_id = 555
    collate = model.make_collate(pad_id, exec_sid, query_sid, image_token_id)
    items = [
        dict(ids=torch.tensor([1, 555, 555, 2, 3]), types=torch.tensor([7, 3, 3, 7, 7]),
             pixel_values=torch.zeros(1, 3, 4, 4), image_grid_thw=torch.tensor([[1, 2, 2]]),
             executed=['move_forward', 'turn_left'], target=3, weight=3.2),
        dict(ids=torch.tensor([1, 555, 2]), types=torch.tensor([7, 3, 7]),
             pixel_values=torch.zeros(1, 3, 4, 4), image_grid_thw=torch.tensor([[1, 2, 2]]),
             executed=[], target=0, weight=1.0),
    ]
    batch = collate(items)
    # sample 0: base 5 + 2 exec + 1 query = 8; sample 1: 3 + 0 + 1 = 4
    assert batch['input_ids'].shape == (2, 8)
    assert batch['attention_mask'].tolist() == [[1] * 8, [1] * 4 + [0] * 4]
    assert batch['exec_index'].tolist() == [[0, 5, 0], [0, 6, 1]], batch['exec_index']
    assert batch['action_index'].tolist() == [[0, 7], [1, 3]]
    assert batch['targets'].tolist() == [3, 0]
    assert abs(batch['weights'][0].item() - 3.2) < 1e-6 and batch['weights'][1].item() == 1.0
    assert batch['input_ids'][0, 5:].tolist() == [901, 902, 900]
    assert batch['input_ids'][1, 3].item() == 900
    assert batch['input_ids'][1, 4:].tolist() == [99] * 4
    assert batch['mm_token_type_ids'][1, 4:].tolist() == [7] * 4


def test_loss_reduction_equivalence():
    """Global weighted mean must equal the DDP-scaled per-rank reduction."""
    torch.manual_seed(0)
    ce = torch.rand(8) + 0.5
    w = torch.rand(8) + 0.5
    world = 4
    shard = 2
    global_mean = (ce * w).sum() / w.sum()
    parts = []
    for r in range(world):
        local_ce, local_w = ce[r * shard:(r + 1) * shard], w[r * shard:(r + 1) * shard]
        w_global = w.sum()  # all-reduced scalar in production
        loss_r = (local_ce * local_w).sum() * world / w_global
        parts.append(loss_r)
    ddp_mean = torch.stack(parts).mean()  # DDP averages gradients of loss_r
    assert torch.allclose(global_mean, ddp_mean, atol=1e-6), (global_mean, ddp_mean)


def test_cosine_warmup():
    assert train.cosine_warmup(0, 1000, 30, 1e-4) == 1e-4 / 30
    assert abs(train.cosine_warmup(30, 1000, 30, 1e-4) - 1e-4) < 1e-9
    assert abs(train.cosine_warmup(1000, 1000, 30, 1e-4)) < 1e-9


def test_sample_index_roundtrip(tmp_path=None):
    import hashlib
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        path = Path(tmp) / 'idx.jsonl'
        with path.open('w') as f:
            f.write(json.dumps([0, 0, 0, 3.2, 100]) + '\n')
            f.write(json.dumps([0, 1, 1, 1.0, 105]) + '\n')
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        samples = data.load_sample_index(path, h, 2)
        assert samples[0] == dict(record_idx=0, t=0, target=0, weight=3.2, est=100)
        try:
            data.load_sample_index(path, '0' * 64, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('hash check missing')


def test_epoch_boundary_advance():
    c = dict(epoch=0, position=100, updates=5, decisions=1000)
    # shard not exhausted: unchanged (mid-shard stop stays resumable)
    assert data.advance_epoch_boundary(c, 200) == c
    # shard exhausted: advance
    c2 = data.advance_epoch_boundary(c, 100)
    assert c2['epoch'] == 1 and c2['position'] == 0 and c2['updates'] == 5
    # boundary resume: position already at shard end -> advance without spin
    c3 = data.advance_epoch_boundary(dict(epoch=0, position=200, updates=9, decisions=9), 200)
    assert c3['epoch'] == 1 and c3['position'] == 0
    # last epoch: reaches protocol epochs (loop exits)
    c4 = data.advance_epoch_boundary(dict(epoch=2, position=50, updates=1, decisions=1), 50)
    assert c4['epoch'] == 3


def test_epoch_boundary_all_ranks_consistent():
    samples = [dict(est=(i * 37) % 500 + 50) for i in range(500)]
    for world in (1, 3, 4):
        plans = data.plan_epoch_batches(samples, 1024, 1109, 0, world)
        assert len({len(x) for x in plans}) == 1
        # identical advance decision on every rank for a shared cursor
        cursor = dict(epoch=0, position=len(plans[0]), updates=1, decisions=1)
        for rank in range(world):
            nxt = data.advance_epoch_boundary(cursor, len(plans[rank]))
            assert nxt['epoch'] == 1 and nxt['position'] == 0


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
