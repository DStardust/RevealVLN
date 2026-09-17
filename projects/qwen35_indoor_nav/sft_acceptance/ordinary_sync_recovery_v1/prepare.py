"""Read old evidence, validate checkpoint on CPU, freeze this bounded repair."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ordinary_baseline_v3'
LINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
import data
import control
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    original = json.loads((OLD / 'PROTOCOL_3GPU_SEG2.json').read_text())
    for name, digest in original['code_sha256'].items():
        assert sha(OLD / name) == digest, 'OLD_CODE_CHANGED:' + name
    assert sha(HERE / 'data.py') == original['code_sha256']['data.py'], 'DATA_COPY_CHANGED'
    ckpt = OLD / 'formal/run_0001/checkpoint_000030400.pt'
    receipt = json.loads(Path(str(ckpt) + '.json').read_text())
    assert sha(ckpt) == receipt['sha256']
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    assert state['cursor'] == receipt['cursor']
    assert state['binding']['sample_index_sha256'] == original['sample_index_sha256']
    assert state['binding']['protocol_sha256'] == sha(OLD / 'PROTOCOL_3GPU_SEG2.json')
    assert all(torch.isfinite(t).all() for t in state['trainable'].values())
    for slot in state['optimizer']['state'].values():
        assert all(torch.isfinite(t).all() for t in slot.values() if isinstance(t, torch.Tensor))
    rows, report = data.load_rows()
    assert report['training_index_sha256'] == original['snapshot']['training_index_sha256']
    samples = data.load_sample_index(OLD / 'SAMPLE_INDEX.jsonl', original['sample_index_sha256'],
                                    original['snapshot']['decisions_per_epoch'])
    plans = [data.plan_epoch_batches(samples, 6144, original['seed'], e, 3) for e in range(3)]
    rank_counts = [control.processed_by_rank(plans, state['cursor'], r) for r in range(3)]
    max_global_batch = max(sum(len(p[r][i]) for r in range(3))
                           for p in plans for i in range(len(p[0])))
    records = [json.loads(s) for s in (OLD / 'formal/run_0001/PROGRESS.jsonl').read_text().splitlines() if s]
    observed_window_decisions = sum(sum(map(sum, r.get('metrics', {}).get('confusion', []))) for r in records)
    highest_legacy_counter = max(r['cumulative_compute']['decisions'] for r in records)
    # Per old checkpoint interval and legacy report gaps: reserve, not measured
    # compute. Ambiguous old counters are not silently relabeled exact usage.
    old_attempt_count = max(1, (OLD / 'formal/supervise_v3.log').read_text().count('torchrun'))
    legacy_reserve = max(20000, original['checkpoint_updates'] * max_global_batch * old_attempt_count)
    initial_charged = max(observed_window_decisions, highest_legacy_counter, sum(rank_counts)) + legacy_reserve
    deadline = min(r['unix'] for r in records) + original['budget']['wall_seconds']
    assert deadline > time.time() + 900, 'NO_WALL_BUDGET'
    assert initial_charged < original['budget']['max_decisions'], 'NO_DECISION_BUDGET'
    restoration = json.loads((OLD / 'formal/lease_3gpu_r1/LEASE_RESULT.json').read_text())
    assert restoration['holders_restored'] and restoration['external_processes_stopped'] == 0
    protocol = copy.deepcopy(original)
    protocol.update(id='Q35N_ORDINARY_SYNC_RECOVERY_V1', created='2026-09-11',
                    log_every_updates=20, max_global_batch_decisions=max_global_batch,
                    parent_protocol_sha256=sha(OLD / 'PROTOCOL_3GPU_SEG2.json'),
                    segment_note='Versioned collective/device/accounting repair; unchanged model definition/data/loss/LR.',
                    accounting=dict(initial_charged_decisions=initial_charged,
                        observed_window_decisions=observed_window_decisions,
                        highest_legacy_counter=highest_legacy_counter,
                        legacy_unobserved_tail='UNKNOWN; conservative budget reservation, not measured usage',
                        legacy_reserve_decisions=legacy_reserve, legacy_attempt_marker_count=old_attempt_count,
                        first_observed_unix=min(r['unix'] for r in records), deadline_unix=deadline),
                    resume_from=dict(path=str(ckpt), sha256=receipt['sha256'],
                                     receipt_sha256=sha(Path(str(ckpt) + '.json')), updates=30400))
    code = ['train.py', 'data.py', 'model.py', 'control.py', 'supervise.py', 'lease_run.py',
            'test_v3.py', 'test_control.py', 'prepare.py', 'monitor.py', 'index.html']
    protocol['code_sha256'] = {name: sha(HERE / name) for name in code}
    for key in ('epochs', 'seed', 'batching', 'optimizer', 'loss', 'snapshot', 'sample_index_sha256',
                'world_size', 'gpus', 'budget', 'data', 'checkpoint_updates', 'kernel'):
        assert protocol[key] == original[key], 'SCIENTIFIC_INVARIANT_CHANGED:' + key
    save('PROTOCOL.json', protocol)
    runbook = json.loads((OLD / 'RUNBOOK_FORMAL_3GPU_R1.json').read_text())
    runbook.update(name='Q35N_ORDINARY_SYNC_RECOVERY_V1', out='lease_v1',
                   lease_wall_seconds=int(deadline - time.time()) + 120,
                   code_sha256={name: sha(HERE / name) for name in code})
    runbook['steps'] = [dict(name='sync_recovery_train', argv=[
        str(LINE.parents[1] / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),
        '-I', '-B', '-u', str(HERE / 'supervise.py')], env_extra={})]
    save('RUNBOOK.json', runbook)
    save('PREFLIGHT.json', dict(status='PASS', unix=time.time(), checkpoint_sha256=sha(ckpt),
         checkpoint_cursor=state['cursor'], rank_plan_decisions=rank_counts,
         total_planned_updates=sum(len(p[0]) for p in plans), max_global_batch_decisions=max_global_batch,
         tensors_finite=True, optimizer_restorable=True, old_restoration=restoration,
         training_index_sha256=report['training_index_sha256'],
         protocol_sha256=sha(HERE / 'PROTOCOL.json'), runbook_sha256=sha(HERE / 'RUNBOOK.json'),
         accounting=protocol['accounting'], scientific_pass=False))
    print(json.dumps(dict(status='PREFLIGHT_PASS', rank_plan_decisions=rank_counts,
                         accounting=protocol['accounting'], max_global_batch=max_global_batch)), flush=True)


if __name__ == '__main__':
    main()
