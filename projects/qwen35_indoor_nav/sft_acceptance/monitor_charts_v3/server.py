"""Read-only chart monitor v3: points at ordinary_baseline_v3 formal run.

Extends sealed monitor_charts_v1 (hashes verified). v3 progress events carry
WINDOWED (per-report, globally all-reduced) metrics, so recent-quality is summed
over recent windows instead of differencing cumulative counters. v6 soft-limit
fields are removed. Read-only: GET /, /api/status, /healthz; everything else 405.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'monitor_charts_v1'
V3 = HERE.parent / 'ordinary_baseline_v3'
EXECUTION = V3 / 'formal'


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


for name, digest in json.loads((BASE / 'CODE_SEAL.json').read_text()).items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, 'FROZEN_MONITOR_CHANGED'
b = load('sealed_charts_v1', BASE / 'server.py')
number = b.number


def chart_points_v3(records, window=20):
    """v3 event schema: CE = metrics.mean_ce (windowed weighted), decisions =
    cumulative_compute.decisions (global), throughput recomputed from a sliding
    window of ~10 events (the cumulative field is inflated across resumes)."""
    points, recent = [], []
    previous = None
    for row in records:
        cursor = row.get('cursor') or {}
        metric = row.get('metrics') or {}
        compute = row.get('cumulative_compute') or {}
        if not all(isinstance(x, dict) for x in (cursor, metric, compute)):
            continue
        unix, decisions = number(row.get('unix')), number(compute.get('decisions'))
        if unix is None or decisions is None:
            continue
        ce = number(metric.get('mean_ce'))
        reset = previous is not None and (unix <= previous[0] or decisions < previous[1])
        if reset:
            recent = []
        if ce is not None:
            recent.append(ce)
            recent = recent[-window:]
        points.append({'unix': unix, 'decisions': decisions, 'updates': number(cursor.get('updates')),
                       'epoch': number(cursor.get('epoch')), 'last_chunk_ce': ce,
                       'rolling_ce': sum(recent) / len(recent) if ce is not None and recent else None,
                       'throughput': None, 'discontinuity': reset})
        previous = (unix, decisions)
    for i, p in enumerate(points):
        j = max(0, i - 10)
        dt = p['unix'] - points[j]['unix']
        dd = p['decisions'] - points[j]['decisions']
        p['throughput'] = dd / dt if dt > 0 and dd >= 0 else None
    return points


def recent_quality_v3(records, min_decisions=1000):
    """Sum v3 windowed confusion matrices covering at least min_decisions global decisions."""
    rows = [r for r in records if isinstance(r, dict)
            and isinstance(r.get('metrics'), dict)
            and isinstance(r['metrics'].get('confusion'), list)
            and isinstance(r.get('cumulative_compute'), dict)
            and type(r['cumulative_compute'].get('decisions')) is int]
    if len(rows) < 2:
        return {'available': False, 'reason': 'INSUFFICIENT_V3_WINDOW_RECORDS'}
    matrix = [[0] * 4 for _ in range(4)]
    total = 0
    ce_sum = 0.0
    for r in reversed(rows[1:]):
        conf = r['metrics']['confusion']
        n = sum(sum(x) for x in conf)
        if n <= 0:
            continue
        for i in range(4):
            for j in range(4):
                matrix[i][j] += conf[i][j]
        total += n
        ce_sum += r['metrics'].get('mean_ce', 0.0) * n
        if total >= min_decisions:
            break
    if total <= 0:
        return {'available': False, 'reason': 'EMPTY_WINDOWS'}
    support = [sum(r) for r in matrix]
    prediction = [sum(matrix[i][j] for i in range(4)) for j in range(4)]
    recall = [matrix[i][i] / support[i] if support[i] else None for i in range(4)]
    accuracy = sum(matrix[i][i] for i in range(4)) / total
    baseline = support[0] / total
    return dict(available=True, decisions=total,
                mean_ce=ce_sum / total, accuracy=accuracy, always_forward_accuracy=baseline,
                accuracy_gain_over_always_forward=accuracy - baseline,
                target_counts=support, prediction_counts=prediction, recall=recall,
                confusion=matrix,
                macro_recall=sum(x for x in recall if x is not None) / max(1, sum(x is not None for x in recall)),
                forward_prediction_fraction=prediction[0] / total,
                stop_predictions=prediction[3], stop_targets=support[3],
                scope='RECENT_ONLINE_FIT_TRAINING_NOT_FIXED_PANEL_NOT_DEV_NOT_NAVIGATION_SUCCESS')


original_collect = b.collect


def collect(*args, **kwargs):
    d = original_collect(*args, execution=EXECUTION, **kwargs)
    progress = d.get('progress', {}).get('data') or {}
    compute = progress.get('cumulative_compute') or {}
    budget = progress.get('budget') or {}
    speed = progress.get('throughput')
    done = compute.get('decisions')
    d['decisions'] = done  # global decisions (cursor.decisions is per-rank in v3)
    d['training_soft_limit'] = None
    limit = budget.get('max_decisions')
    d['decision_limit'] = limit
    d['charged_compute_decisions'] = done
    if None not in (done, speed, limit) and speed > 0:
        d['budget_eta_seconds'] = max(0, limit - done) / speed
        total = d.get('epoch_total_decisions')
        if total:
            d['epoch_eta_seconds'] = max(0, total - (done % max(1, total))) / speed
    lease = b.read_json(EXECUTION.parent / 'lease_3gpu/LEASE_RESULT.json', b.LINE)
    d['v3_lease'] = lease
    records = b.tail_records(EXECUTION / 'run_0001/PROGRESS.jsonl')['records']
    d['points'] = chart_points_v3(records)
    # steady throughput from the recent window; fixes the resume-inflated field
    pts = [p for p in d['points'] if p.get('throughput') is not None and not p.get('discontinuity')]
    if pts:
        steady = pts[-1]['throughput']
        for key in ('budget_eta_seconds', 'epoch_eta_seconds', 'estimated_segment_remaining_seconds'):
            pass  # recomputed below where possible
        limit = d.get('decision_limit')
        done = d.get('decisions')
        total = d.get('epoch_total_decisions')
        if None not in (done, limit) and steady > 0:
            d['budget_eta_seconds'] = max(0, limit - done) / steady
        if total and done is not None and steady > 0:
            d['epoch_eta_seconds'] = max(0, total - (done % max(1, total))) / steady
        d['steady_decisions_per_second'] = steady
    d['recent_quality'] = recent_quality_v3(records, 1000)
    d['monitor_version'] = 'v3'
    return d


b.collect = collect
b.EXECUTION = EXECUTION

if __name__ == '__main__':
    for name, digest in json.loads((HERE / 'CODE_SEAL.json').read_text()).items():
        path = HERE / name
        assert path.resolve().is_relative_to(b.LINE) and \
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
    b.main()
