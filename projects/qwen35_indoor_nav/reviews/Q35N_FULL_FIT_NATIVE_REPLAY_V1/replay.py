"""Replay only deployable observations; reference logits are never read here."""
import hashlib
import json
from pathlib import Path
import time


def validate_input(row):
    assert set(row) == {'record_id', 'instruction', 'rgb_sha256', 'executed_actions'}
    assert isinstance(row['instruction'], str) and row['instruction']
    assert 1 <= len(row['rgb_sha256']) <= 2 and len(row['executed_actions']) <= 8
    assert all(x in ('move_forward', 'turn_left', 'turn_right') for x in row['executed_actions'])
    assert all(isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)
               for x in [row['record_id']] + row['rgb_sha256'])


def run(model, policy, forward, c, fingerprint, initial_sha):
    import torch
    from PIL import Image
    plan = c.read(c.HERE / 'REPLAY_PROTOCOL.json')
    assert plan['runtime_allowed'] and plan['forward_decisions'] == 5519
    assert initial_sha == c.read(c.LINE / 'closed_loop_bench/ordinary_stop_calibration_fit_v2/run_001/MODEL_LOADED.json')['trainable_sha256']
    source = c.LINE / 'sft_acceptance/ordinary_stop_row_v12/POLICY_INPUTS.jsonl'
    rows = [json.loads(x) for x in source.read_text().splitlines()]
    assert len(rows) == 5487 and len({r['record_id'] for r in rows}) == 5487
    data = c.LINE / 'data_pipeline/ordinary_route_teacher_v11/run_001/content'
    for row in rows:
        validate_input(row)

    class Store:
        def get(self, index, t):
            assert t == 0 and 0 <= index < len(rows)
            row = rows[index]
            raw = []
            for digest in row['rgb_sha256']:
                with Image.open(data / (digest + '.png')) as image:
                    rgb = image.convert('RGB')
                    assert rgb.size == (224, 224)
                    value = rgb.tobytes()
                assert hashlib.sha256(value).hexdigest() == digest
                raw.append(value)
            window = c.Window()
            window.instruction = row['instruction']
            window.executed = list(row['executed_actions'])
            window.images = raw
            return window.item()

    samples = [dict(record_idx=i, t=0, target=0, weight=1.) for i in range(len(rows))]
    dataset = model.DecisionDataset(samples, Store(), policy.processor)
    output = c.HERE / 'run_001'
    began = time.monotonic()
    with (output / 'REPLAY_LOGITS.jsonl').open('x') as stream:
        with torch.inference_mode():
            for i, row in enumerate(rows):
                # Same creation context as the actual native online inference loop.
                encoded = dataset[i]
                logits = forward([encoded]).float().cpu()
                assert logits.shape == (1, 4) and torch.isfinite(logits).all()
                stream.write(json.dumps(dict(index=i, record_id=row['record_id'], logits=logits[0].tolist())) + '\n')
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    stream.flush()
                    value = dict(status='REPLAYING', unix=time.time(), completed=i + 1,
                                 total=5487, total_forward_decisions=i + 1 + 32,
                                 wall_seconds=time.monotonic() - began, optimizer_updates=0, simulator_actions=0)
                    c.write(output / 'PROGRESS.json', value)
                    print(json.dumps(value), flush=True)
    assert fingerprint() == initial_sha
    c.write(output / 'REPLAY_RESULT.json', dict(status='COMPLETE_PENDING_CPU_PARITY', unix=time.time(),
        inputs=5487, fixture_forward_decisions=32, replay_forward_decisions=5487,
        total_forward_decisions=5519, backward_calls=0, optimizer_updates=0, simulator_actions=0,
        trainable_fingerprint=initial_sha, model_unchanged=True,
        output_sha256=c.sha(output / 'REPLAY_LOGITS.jsonl'),
        wall_seconds=time.monotonic() - began, navigation_gain_verified=False), True)
